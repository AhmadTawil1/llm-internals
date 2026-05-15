import math
from typing import Literal, cast, overload

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinygpt.config import TinyGPTConfig
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
    attn_weights = F.softmax(scores, dim=-1)

    # NaN replacement for fully masked pad rows.
    # If a query is a pad token, its entire row is masked to -inf, causing softmax
    # to yield NaN. We replace these NaNs with 0.0 because the loss ignores pad tokens.
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # 4. Weighted sum of values.
    #    (..., T, T) @ (..., T, d_v) -> (..., T, d_v)
    out = torch.matmul(attn_weights, v)

    return out, attn_weights



class SingleHeadAttention(nn.Module):
    """Single-head attention.

    Pedagogical reference; see MultiHeadAttention for the full implementation.
    """

    causal_mask: torch.Tensor

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.d_model = config.d_model

        # Three independent linear projections. bias=False is the modern default
        # (LLaMA, Qwen, Mistral all drop the bias on Q/K/V).
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Pre-build causal mask up to max_seq_len; slice at forward time.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        assert D == self.d_model, f"expected d_model={self.d_model}, got {D}"

        q = self.W_q(x)  # (B, T, D)
        k = self.W_k(x)  # (B, T, D)
        v = self.W_v(x)  # (B, T, D)

        # Add a singleton head dim so we can reuse the same code as multi-head.
        # (B, T, D) -> (B, 1, T, D)
        q, k, v = (t.unsqueeze(1) for t in (q, k, v))

        mask = self.causal_mask[:, :, :T, :T]  # (1, 1, T, T)
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        out = self.dropout(out)

        # Drop the head dim and project.
        out = out.squeeze(1)  # (B, T, D)
        out = cast(torch.Tensor, self.W_o(out))
        return out


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with RoPE and causal masking."""

    causal_mask: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
        self,
        config: TinyGPTConfig,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.n_heads
        self.d_head = config.head_dim

        # One fused projection for Q, K, V is faster (one matmul instead of three),
        # but three separate ones are easier to read.
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
            persistent=False,
        )

        cos, sin = precompute_freqs_cis(self.d_head, config.max_seq_len, base=config.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, d_h) -> (B, T, D)"""
        B, H, T, d_h = x.shape
        x = x.transpose(1, 2).contiguous()  # (B, T, H, d_h)
        return x.view(B, T, H * d_h)  # (B, T, D)

    @overload
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_weights: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_weights: Literal[True] = True,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        B, T, D = x.shape
        assert D == self.d_model

        # (B, T, D) -> (B, T, H, d_h)
        q = self.W_q(x).view(B, T, self.num_heads, self.d_head)
        k = self.W_k(x).view(B, T, self.num_heads, self.d_head)
        v = self.W_v(x).view(B, T, self.num_heads, self.d_head)

        # (B, T, H, d_h) — shape preserved, rotations applied positionally
        q, k = apply_rotary_emb(q, k, self.cos[:T], self.sin[:T])

        # (B, T, H, d_h) -> (B, H, T, d_h)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Slice the causal mask to the actual seq length.
        mask = self.causal_mask[:, :, :T, :T]  # (1, 1, T, T) — broadcasts over (B, H)

        # Pad mask
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T) so it broadcasts over heads and query positions
            pad_mask = attention_mask[:, None, None, :]
            # Combine causal and pad masks
            mask = torch.logical_and(mask, pad_mask)

        # out: (B, H, T, d_h)  attn_weights: (B, H, T, T)
        out, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        out = self.attn_dropout(out)  # (B, H, T, d_h)

        # Merge heads and project.
        out = self._merge_heads(out)  # (B, T, D)
        out = cast(torch.Tensor, self.resid_dropout(self.W_o(out)))  # (B, T, D)

        if return_weights:
            return out, attn_weights
        return out
