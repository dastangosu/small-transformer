# pyright: reportMissingImports=false, reportInvalidTypeForm=false

import torch

try:
	import triton
	import triton.language as tl

	_TRITON_AVAILABLE = True
	_TRITON_IMPORT_ERROR = None
except Exception as exc:  
	triton = None
	tl = None
	_TRITON_AVAILABLE = False
	_TRITON_IMPORT_ERROR = exc


if _TRITON_AVAILABLE:

	@triton.jit
	def _flash_fwd_inner(
		acc,
		l_i,
		m_i,
		q,
		K_block_ptr,
		V_block_ptr,
		block_index_q,
		softmax_scale,
		BLOCK_M: tl.constexpr,
		BLOCK_N: tl.constexpr,
		STAGE: tl.constexpr,
		offs_q: tl.constexpr,
		offs_n: tl.constexpr,
		SEQ_LEN: tl.constexpr,
	):
		if STAGE == 1:
			lo, hi = 0, block_index_q * BLOCK_M
		elif STAGE == 2:
			lo, hi = block_index_q * BLOCK_M, (block_index_q + 1) * BLOCK_M
			lo = tl.multiple_of(lo, BLOCK_M)
		else:
			lo, hi = 0, SEQ_LEN

		K_block_ptr = tl.advance(K_block_ptr, (0, lo))
		V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

		for start_n in range(lo, hi, BLOCK_N):
			start_n = tl.multiple_of(start_n, BLOCK_N)
			valid_n = (start_n + offs_n) < SEQ_LEN

			k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
			qk = tl.dot(q, k) * softmax_scale

			if STAGE == 2:
				causal_mask = offs_q[:, None] >= (start_n + offs_n[None, :])
				qk = tl.where(causal_mask & valid_n[None, :], qk, -1.0e6)
			else:
				qk = tl.where(valid_n[None, :], qk, -1.0e6)

			m_ij = tl.maximum(m_i, tl.max(qk, 1))
			qk = qk - m_ij[:, None]

			p = tl.math.exp(qk)
			l_ij = tl.sum(p, 1)

			alpha = tl.math.exp(m_i - m_ij)
			l_i = l_i * alpha + l_ij

			v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
			p = p.to(tl.float16)
			acc = acc * alpha[:, None]
			acc = tl.dot(p, v, acc)

			m_i = m_ij
			V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
			K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

		return acc, l_i, m_i


	@triton.jit
	def _flash_fwd(
		Q,
		K,
		V,
		softmax_scale,
		LSE,
		O,
		stride_q_batch,
		stride_q_head,
		stride_q_seq,
		stride_q_dim,
		stride_k_batch,
		stride_k_head,
		stride_k_seq,
		stride_k_dim,
		stride_v_batch,
		stride_v_head,
		stride_v_seq,
		stride_v_dim,
		stride_o_batch,
		stride_o_head,
		stride_o_seq,
		stride_o_dim,
		BATCH_SIZE,
		NUM_HEADS: tl.constexpr,
		SEQ_LEN: tl.constexpr,
		HEAD_DIM: tl.constexpr,
		BLOCK_M: tl.constexpr,
		BLOCK_N: tl.constexpr,
		STAGE: tl.constexpr,
	):
		block_index_q = tl.program_id(0)
		index_batch_head = tl.program_id(1)
		index_batch = index_batch_head // NUM_HEADS
		index_head = index_batch_head % NUM_HEADS

		q_offset = (
			index_batch.to(tl.int64) * stride_q_batch
			+ index_head.to(tl.int64) * stride_q_head
		)
		k_offset = (
			index_batch.to(tl.int64) * stride_k_batch
			+ index_head.to(tl.int64) * stride_k_head
		)
		v_offset = (
			index_batch.to(tl.int64) * stride_v_batch
			+ index_head.to(tl.int64) * stride_v_head
		)
		o_offset = (
			index_batch.to(tl.int64) * stride_o_batch
			+ index_head.to(tl.int64) * stride_o_head
		)

		Q_block_ptr = tl.make_block_ptr(
			base=Q + q_offset,
			shape=(SEQ_LEN, HEAD_DIM),
			strides=(stride_q_seq, stride_q_dim),
			offsets=(block_index_q * BLOCK_M, 0),
			block_shape=(BLOCK_M, HEAD_DIM),
			order=(1, 0),
		)
		K_block_ptr = tl.make_block_ptr(
			base=K + k_offset,
			shape=(HEAD_DIM, SEQ_LEN),
			strides=(stride_k_dim, stride_k_seq),
			offsets=(0, 0),
			block_shape=(HEAD_DIM, BLOCK_N),
			order=(0, 1),
		)
		V_block_ptr = tl.make_block_ptr(
			base=V + v_offset,
			shape=(SEQ_LEN, HEAD_DIM),
			strides=(stride_v_seq, stride_v_dim),
			offsets=(0, 0),
			block_shape=(BLOCK_N, HEAD_DIM),
			order=(1, 0),
		)
		O_block_ptr = tl.make_block_ptr(
			base=O + o_offset,
			shape=(SEQ_LEN, HEAD_DIM),
			strides=(stride_o_seq, stride_o_dim),
			offsets=(block_index_q * BLOCK_M, 0),
			block_shape=(BLOCK_M, HEAD_DIM),
			order=(1, 0),
		)

		offs_q = block_index_q * BLOCK_M + tl.arange(0, BLOCK_M)
		offs_n = tl.arange(0, BLOCK_N)

		m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
		l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
		acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

		q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

		if STAGE == 1 or STAGE == 3:
			acc, l_i, m_i = _flash_fwd_inner(
				acc,
				l_i,
				m_i,
				q,
				K_block_ptr,
				V_block_ptr,
				block_index_q,
				softmax_scale,
				BLOCK_M,
				BLOCK_N,
				4 - STAGE,
				offs_q,
				offs_n,
				SEQ_LEN,
			)
		if STAGE == 3:
			acc, l_i, m_i = _flash_fwd_inner(
				acc,
				l_i,
				m_i,
				q,
				K_block_ptr,
				V_block_ptr,
				block_index_q,
				softmax_scale,
				BLOCK_M,
				BLOCK_N,
				2,
				offs_q,
				offs_n,
				SEQ_LEN,
			)

		m_i = m_i + tl.math.log(l_i)
		acc = acc / l_i[:, None]

		lse_ptrs = LSE + index_batch_head * SEQ_LEN + offs_q
		tl.store(lse_ptrs, m_i, mask=offs_q < SEQ_LEN)
		tl.store(O_block_ptr, acc.to(O.type.element_ty), boundary_check=(0, 1))


def _naive_attention(
	Q: torch.Tensor,
	K: torch.Tensor,
	V: torch.Tensor,
	causal: bool,
	softmax_scale: float,
) -> torch.Tensor:
	scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
	if causal:
		T = Q.shape[-2]
		mask = torch.tril(torch.ones((T, T), device=Q.device, dtype=torch.bool))
		scores = scores.masked_fill(~mask, float("-inf"))
	probs = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
	return torch.matmul(probs, V)


def is_triton_flash_attention_available() -> bool:
	return _TRITON_AVAILABLE and torch.cuda.is_available()


def _triton_forward(
	Q: torch.Tensor,
	K: torch.Tensor,
	V: torch.Tensor,
	causal: bool,
	softmax_scale: float,
) -> torch.Tensor:
	if not _TRITON_AVAILABLE:
		raise RuntimeError("Triton is not available") from _TRITON_IMPORT_ERROR
	if Q.device.type != "cuda":
		raise ValueError("Triton FlashAttention requires CUDA tensors")
	if Q.shape != K.shape or Q.shape != V.shape:
		raise ValueError("Q, K, V must all have shape (B, H, T, D)")

	B, H, T, D = Q.shape
	if D not in (16, 32, 64, 128):
		raise ValueError("This minimal kernel supports head_dim in {16, 32, 64, 128}")

	O = torch.empty_like(Q)
	lse = torch.empty((B, H, T), device=Q.device, dtype=torch.float32)
	stage = 3 if causal else 1
	BLOCK_M = 64
	BLOCK_N = 64

	grid = (triton.cdiv(T, BLOCK_M), B * H, 1)
	_flash_fwd[grid](
		Q=Q,
		K=K,
		V=V,
		softmax_scale=softmax_scale,
		LSE=lse,
		O=O,
		stride_q_batch=Q.stride(0),
		stride_q_head=Q.stride(1),
		stride_q_seq=Q.stride(2),
		stride_q_dim=Q.stride(3),
		stride_k_batch=K.stride(0),
		stride_k_head=K.stride(1),
		stride_k_seq=K.stride(2),
		stride_k_dim=K.stride(3),
		stride_v_batch=V.stride(0),
		stride_v_head=V.stride(1),
		stride_v_seq=V.stride(2),
		stride_v_dim=V.stride(3),
		stride_o_batch=O.stride(0),
		stride_o_head=O.stride(1),
		stride_o_seq=O.stride(2),
		stride_o_dim=O.stride(3),
		BATCH_SIZE=B,
		NUM_HEADS=H,
		SEQ_LEN=T,
		HEAD_DIM=D,
		BLOCK_M=BLOCK_M,
		BLOCK_N=BLOCK_N,
		STAGE=stage,
		num_warps=4,
		num_stages=2,
	)
	return O


class _TritonFlashAttentionFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, Q, K, V, causal, softmax_scale):
		ctx.causal = causal
		ctx.softmax_scale = softmax_scale
		ctx.save_for_backward(Q, K, V)
		return _triton_forward(Q, K, V, causal, softmax_scale)

	@staticmethod
	def backward(ctx, dO):
		Q, K, V = ctx.saved_tensors
		with torch.enable_grad():
			q = Q.detach().requires_grad_(True)
			k = K.detach().requires_grad_(True)
			v = V.detach().requires_grad_(True)
			out = _naive_attention(q, k, v, ctx.causal, ctx.softmax_scale)
			dQ, dK, dV = torch.autograd.grad(
				out,
				(q, k, v),
				grad_outputs=dO,
				allow_unused=False,
			)
		return dQ, dK, dV, None, None


def flash_attention_forward(
	Q: torch.Tensor,
	K: torch.Tensor,
	V: torch.Tensor,
	causal: bool = True,
	softmax_scale: float = None,
	force_naive_fallback: bool = False,
) -> torch.Tensor:
	if softmax_scale is None:
		softmax_scale = Q.shape[-1] ** -0.5

	if force_naive_fallback or not is_triton_flash_attention_available():
		return _naive_attention(Q, K, V, causal=causal, softmax_scale=softmax_scale)

	return _TritonFlashAttentionFunction.apply(
		Q.contiguous(),
		K.contiguous(),
		V.contiguous(),
		causal,
		softmax_scale,
	)
