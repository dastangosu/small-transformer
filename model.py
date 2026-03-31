import torch
import torch.nn as nn 
from torch.optim import Optimizer
import math
import os

from flash_attention_triton import (
    flash_attention_forward,
    is_triton_flash_attention_available,
)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        sigma = (2/(in_features+out_features))**0.5
        self.W = nn.Parameter(torch.empty(out_features,in_features))
        torch.nn.init.trunc_normal_(self.W,mean=0,std=sigma,a = -3*sigma, b= 3*sigma)
    
    def forward(self, x: torch.Tensor):
        return x @ self.W.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype = None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_embeddings,embedding_dim))
        torch.nn.init.trunc_normal_(self.W,mean=0.0,std=1.0,a = -3.0, b= 3.0)

    def forward(self, token_ids: torch.Tensor):
        return self.W[token_ids]
        


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor):  # (B,T,C)
        x_sqr = x**2
        x_sqr_mean = x_sqr.mean(-1,keepdim=True) # (B,T,1)
        rms = (x_sqr_mean + self.eps)**0.5
        rmsnorm = (x/rms) * self.gain # (B,T,C) * (1,1,C)
        return rmsnorm
    


class FFN_swiglu(nn.Module):
    def __init__(self,d_model:int, d_ff:int):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear3 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):
        r1 = self.linear3(x)    #(B,T,d_ff)
        r2 = self.linear1(x)    #(B,T,d_ff)
        r2 = r2 * torch.sigmoid(r2)    
        r3 = r1 * r2
        return self.linear2(r3)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        i = torch.arange(max_seq_len).unsqueeze(1) # (max_seq_len,1)
        k = torch.arange(1,d_k//2+1)    #(d_k/2,)
        exponent = (2*k-2) / d_k        #(d_k/2,)
        denom = theta ** (exponent)     #(d_k/2,)
        angle = i/denom     #(max_seq_len,1) / (1, d_k/2) = (max_seq_len,d_k/2)        
        self.register_buffer("sin", torch.sin(angle))
        self.register_buffer("cos", torch.cos(angle))

    def forward(self,x: torch.Tensor, token_positions: torch.Tensor  = None):
        #x -> (B,T,dk)   positions -> (B,T)
        T = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(T, device=x.device)[None, :]
        sines = self.sin[token_positions] #(B,T,d_k/2)
        cosines = self.cos[token_positions] # (B,T,d_k/2)
        even = x[...,::2]   #(B,T,d_k/2)
        odd = x[...,1::2]   #(B,T,d_k/2)
        t1 = even * cosines - odd * sines   #(B,T,d_k/2)
        t2 = even * sines + odd * cosines    #(B,T,d_k/2)
        stacked = torch.stack((t1,t2),dim=-1) # (B,T,d_k/2,2)
        return stacked.reshape(*x.shape)    #(B,T,d_k)


    

def softmax(t: torch.Tensor, dim:int):
    t_max = torch.max(t,dim = dim, keepdim=True).values
    t = t - t_max 
    exp = torch.exp(t)
    exp_sum = torch.sum(exp,dim = dim, keepdim=True)
    return exp/exp_sum

def scaled_dot_product_attention(Q: torch.Tensor, K:torch.Tensor, V: torch.Tensor, mask: torch.Tensor ):
    scale = K.shape[-1] ** -0.5
    alpha = (Q @ K.transpose(-2,-1)) * scale        #(B,T,T), T = seq_len
    alpha = alpha.masked_fill(mask == False,float('-inf'))
    alpha = softmax(alpha, dim=-1)
    return alpha @ V
    

class MHA_with_RoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attention_impl: str = "naive"):
        super().__init__()
        self.query = Linear(d_model,d_model)
        self.key = Linear(d_model,d_model)
        self.value  = Linear(d_model,d_model)
        self.proj = Linear(d_model,d_model)
        self.num_heads = num_heads
        if attention_impl not in {"naive", "triton"}:
            raise ValueError("attention_impl must be one of: naive, triton")
        self.attention_impl = attention_impl
        self._attention_debug = os.getenv("ATTENTION_DEBUG", "0") == "1"
        self._attention_debug_printed = False
        self._rope = None
        self._rope_theta = None
        self._rope_max_seq_len = None

    @staticmethod
    def _naive_causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        T = Q.shape[-2]
        tril = torch.tril(torch.ones(T, T, device=Q.device))
        mask = tril == 1
        return scaled_dot_product_attention(Q, K, V, mask)

    def forward(self,x:torch.Tensor, theta: float, max_seq_len: int, token_positions:torch.Tensor = None):
        B,T,d_model = x.shape
        num_heads = self.num_heads
        Q = self.query(x)       #B,T,d_model
        K = self.key(x)
        V = self.value(x)

        if (
            self._rope is None
            or self._rope_theta != theta
            or self._rope_max_seq_len != max_seq_len
            or self._rope.sin.device != x.device
        ):
            # Keep RoPE cache out of state_dict (it is a runtime cache, not trainable state)
            object.__setattr__(self, "_rope", RoPE(theta, d_model//num_heads, max_seq_len).to(x.device))
            self._rope_theta = theta
            self._rope_max_seq_len = max_seq_len

        rope = self._rope
        Q = Q.reshape(B,T,num_heads,d_model//num_heads).transpose(1,2)
        K = K.reshape(B,T,num_heads,d_model//num_heads).transpose(1,2)
        V = V.reshape(B,T,num_heads,d_model//num_heads).transpose(1,2)
        Q = rope(Q,token_positions)
        K = rope(K,token_positions)

        use_triton = (
            self.attention_impl == "triton"
            and is_triton_flash_attention_available()
            and x.is_cuda
        )

        if use_triton:
            input_dtype = Q.dtype
            if input_dtype != torch.float16:
                Q = Q.to(torch.float16)
                K = K.to(torch.float16)
                V = V.to(torch.float16)
            attention = flash_attention_forward(Q, K, V, causal=True)
            if attention.dtype != input_dtype:
                attention = attention.to(input_dtype)
        else:
            attention = self._naive_causal_attention(Q, K, V)

        if self._attention_debug and not self._attention_debug_printed:
            backend = "triton" if use_triton else "naive-fallback"
            print(f"[MHA_with_RoPE] attention backend={backend}")
            self._attention_debug_printed = True

        attention = attention.transpose(1,2).reshape(B,T,d_model)
        return self.proj(attention)
    
class TransformerBlock(nn.Module):
    def __init__(self,d_model:int, num_heads: int, d_ff: int, attention_impl: str = "naive"):
        super().__init__()
        self.mha = MHA_with_RoPE(d_model, num_heads, attention_impl=attention_impl)
        self.ffn = FFN_swiglu(d_model, d_ff)
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor,theta,max_seq_length):
        x = x + self.mha(self.rmsnorm1(x),theta,max_seq_length)
        x = x + self.ffn(self.rmsnorm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, attention_impl: str = "naive"):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    attention_impl=attention_impl,
                )
                for _ in range(num_layers)
            ]
        )
        self.rmsnorm = RMSNorm(d_model)
        self.lmhead = Linear(d_model,vocab_size)
        self.context_length = context_length

    def forward(self, x, theta):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, theta, self.context_length)
        x = self.rmsnorm(x)
        x = self.lmhead(x)
        return x
    
def CrossEntropy(logits: torch.Tensor,targets: torch.Tensor):    #logits: B,T,vocab_size     targets: B,T
    C = logits.shape[-1]
    logits = logits.reshape(-1,C)
    targets = targets.reshape(-1)
    log_probs = log_softmax(logits,dim=-1)
    rows = torch.arange(logits.shape[0], device=logits.device)
    return -log_probs[rows,targets].mean()


def log_softmax(t: torch.Tensor, dim:int):
    t_max = torch.max(t,dim = dim, keepdim=True).values
    t = t - t_max
    exp = torch.exp(t)
    exp_sum = torch.sum(exp,dim = dim, keepdim=True)
    log_exp_sum = torch.log(exp_sum)
    return t - log_exp_sum


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m = state["m"]
                v = state["v"]
                t = state["t"] + 1  
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                with torch.no_grad():
                    p -= lr * m_hat / (torch.sqrt(v_hat) + eps)
                    if weight_decay != 0:
                        p -= lr * weight_decay * p

                state["t"] = t

        return loss
    

def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    # warmup
    if t < T_w:
        return (t / T_w) * alpha_max
    # cosine annealing
    elif T_w <= t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        cosine = math.cos(math.pi * progress)
        return alpha_min + 0.5 * (1 + cosine) * (alpha_max - alpha_min)
    #post annealing
    else:
        return alpha_min



def gradient_clipping(parameters, max_norm, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return
    # l2 norm
    total_norm = torch.sqrt(
        sum(torch.sum(g ** 2) for g in grads)
    )
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)

        for g in grads:
            g.mul_(scale)

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer=None):
    checkpoint = torch.load(src, map_location=next(model.parameters()).device)

    # backward compatibility: older checkpoints may include cached RoPE buffers
    model_state = {
        k: v
        for k, v in checkpoint["model_state"].items()
        if not (k.endswith("._rope.sin") or k.endswith("._rope.cos"))
    }

    model.load_state_dict(model_state)

    # optimizer state is optional (e.g. inference-only checkpoints)
    if optimizer is not None:
        if "optimizer_state" not in checkpoint:
            raise KeyError("Checkpoint does not contain 'optimizer_state'")
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint.get("iteration", 0)




