import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinygpt.rope import apply_rotary_emb, precompute_freqs_cis


def scaled_dot_product_attention(
    q: torch.Tensor,  # (..., T, d_k)
    k: torch.Tensor,  # (..., T, d_k)
    v: torch.Tensor,  # (..., T, d_v)
    mask: torch.Tensor | None = None,  # broadcastable to (..., T, T); True/1 = keep, False/0 = mask
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (output, attn_weights) where output has shape (..., T, d_v)
    and attn_weights has shape (..., T, T).

    The leading dims (...) can be (B,) for single-head or (B, H) for multi-head.
    matmul broadcasts over them — that is the entire trick.
    """
    d_k = q.size(-1)

    # 1. Score: every query dotted with every key.
    #    (..., T, d_k) @ (..., d_k, T) -> (..., T, T)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. Mask: positions where mask == 0 get -inf, so softmax sends them to 0.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 3. Softmax over the LAST axis (the "key" axis).
    #    Each row of the (T, T) score matrix becomes a probability distribution.
    attn = F.softmax(scores, dim=-1)

    # 4. Weighted sum of values.
    #    (..., T, T) @ (..., T, d_v) -> (..., T, d_v)
    output = torch.matmul(attn, v)

    return output, attn


def build_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Returns a (1, 1, T, T) lower-triangular mask of 1s and 0s.
    Leading singleton dims let it broadcast over (B, H).
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)


class SingleHeadAttention(nn.Module):
    """Single-head attention. Throwaway — you'll replace with MultiHeadAttention.
    Implementing it first keeps the moving parts visible."""

    causal_mask: torch.Tensor

    def __init__(self, d_model: int, max_seq_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model

        # Three independent linear projections. bias=False is the modern default
        # (LLaMA, Qwen, Mistral all drop the bias on Q/K/V).
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Pre-build causal mask up to max_seq_len; slice at forward time.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        assert D == self.d_model, f"expected d_model={self.d_model}, got {D}"

        q = self.W_q(x)  # (B, T, D)
        k = self.W_k(x)
        v = self.W_v(x)

        # Add a singleton head dim so we can reuse the same code as multi-head.
        # (B, T, D) -> (B, 1, T, D)
        q, k, v = (t.unsqueeze(1) for t in (q, k, v))

        mask = self.causal_mask[:, :, :T, :T]  # (1, 1, T, T)
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        out = self.dropout(out)

        # Drop the head dim and project.
        out = out.squeeze(1)  # (B, T, D)
        out = self.W_o(out)
        return out  # type: ignore[no-any-return]


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention. The version you'll actually use."""

    causal_mask: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # One fused projection for Q, K, V is faster (one matmul instead of three),
        # but three separate ones are easier to read. Stick with three for Project 0.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len),
            persistent=False,
        )

        cos, sin = precompute_freqs_cis(self.d_head, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, H, T, d_h)"""
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.d_head)  # (B, T, H, d_h)
        return x.transpose(1, 2)  # (B, H, T, d_h)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, d_h) -> (B, T, D)"""
        B, H, T, d_h = x.shape
        x = x.transpose(1, 2).contiguous()  # (B, T, H, d_h)
        return x.view(B, T, H * d_h)  # (B, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        assert D == self.d_model

        # Project to (B, T, H, d_h)
        q = self.W_q(x).view(B, T, self.num_heads, self.d_head)
        k = self.W_k(x).view(B, T, self.num_heads, self.d_head)
        v = self.W_v(x).view(B, T, self.num_heads, self.d_head)

        # Apply RoPE to queries and keys
        q, k = apply_rotary_emb(q, k, self.cos[:T], self.sin[:T])

        # Transpose to (B, H, T, d_h) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Slice the causal mask to the actual seq length.
        mask = self.causal_mask[:, :, :T, :T]  # (1, 1, T, T) — broadcasts over (B, H)

        # Reuse the function from Pass 1.
        out, attn = scaled_dot_product_attention(q, k, v, mask)
        out = self.attn_dropout(out)  # (B, H, T, d_h)

        # Merge heads and project.
        out = self._merge_heads(out)  # (B, T, D)
        out = self.resid_dropout(self.W_o(out))  # (B, T, D)
        return out  # type: ignore[no-any-return]
