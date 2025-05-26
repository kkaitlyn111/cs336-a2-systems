import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn
from einops import einsum, rearrange
import numpy as np
from jaxtyping import Float, Int
import numpy.typing as npt
from torch import Tensor, LongTensor

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                max_seq_len: int = None,
                theta: float = None,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None):
        super().__init__();
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.device = device
        self.dtype = dtype

        self.theta = theta
        self.max_seq_len = max_seq_len

        eps: float = 1e-5

        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype, apply_rope=True)
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, device=device, dtype=dtype)

    @nvtx.range("TransformerBlock forward pass")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm attention
        with nvtx.range("attention block"):
            h = self.ln1(x)
            h = self.attn(h)
            x = x + h
        
        # pre-norm feedforward
        with nvtx.range("ffn block"):
            h = self.ln2(x)
            h = self.ffn(h)
            x = x + h
        
        return x


class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int,
                max_seq_len: int = None,
                theta: float = None,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.device = device
        self.dtype = dtype

        self.theta = theta
        # Use context_length as max_seq_len if not provided
        self.max_seq_len = max_seq_len if max_seq_len is not None else context_length

        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, self.max_seq_len, theta, device, dtype))

        self.ln_final = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        # Note: weight shape is transposed compared to the test's expectations
        self.lm_head = Linear(vocab_size, d_model, device, dtype)  # [10000, 64]

    @nvtx.range("LM forward pass")
    def forward(self, x: torch.Tensor):
        # token embeddings
        with nvtx.range("token embeddings"):
            x = self.token_embeddings(x)
        
        # apply transformer layers
        with nvtx.range("apply transformer layers"):
            for i in range(self.num_layers):
                x = self.layers[i](x)
        
        # final layer norm
        with nvtx.range("final layer norm"):
            x = self.ln_final(x)
        
        with nvtx.range("LM head"):
            x = self.lm_head(x)
        
        return x


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None, apply_rope: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.theta = theta
        assert d_model % num_heads == 0
        self.d_v = d_model // num_heads
        self.d_k = self.d_v

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is not None and apply_rope:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    @nvtx.range("MultiheadSelfAttention forward pass")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # project queries, keys, and values
        with nvtx.range("qkv projection"):
            Q_x = self.q_proj(x)
            K_x = self.k_proj(x)
            V_x = self.v_proj(x)

        # reshape for multi-head attention
            Q = rearrange(Q_x, "b s (n_h d_k) -> b n_h s d_k", n_h=self.num_heads)
            K = rearrange(K_x, "b s (n_h d_k) -> b n_h s d_k", n_h=self.num_heads)
            V = rearrange(V_x, "b s (n_h d_v) -> b n_h s d_v", n_h=self.num_heads)

        # apply RoPE if enabled
        with ntvx.range("apply rope"):
            if self.rope is not None:
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
                Q = self.rope(Q, positions)
                K = self.rope(K, positions)
            
        with ntvx.range("attention mechanism"):
            # creating attention mask here
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).to(torch.bool)
            mask = ~mask 
            mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)

            # apply attention
            result = scaled_dot_product_attention(Q, K, V, mask=mask)

        # reshape back and project
        with ntvx.range("output projection"):
            result = rearrange(result, "b n_h s d_k -> b s (n_h d_k)")
            result = self.output_proj(result)
        
        return result

@nvtx.range("scaled dot product attention")
def scaled_dot_product_attention(Q: Float[Tensor, " ... queries d_k"], K: Float[Tensor, " ... keys d_k"], V: Float[Tensor, " ... values d_v"], mask: Float[Tensor, " ... queries keys"] | None = None,) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    with ntvx.range("attention scores"):
        attention_scores = torch.einsum("...qd,...kd->...qk", Q, K)
        attention_scores = attention_scores / np.sqrt(d_k)

    with ntvx.range("masking"):
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == False, -1e9)
    
    with ntvx.range("softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    
    with ntvx.range("weighted sum"):
        output = torch.einsum("...qk,...kd->...qd", attention_weights, V)
    
    return output

    