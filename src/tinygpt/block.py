import torch
import torch.nn as nn

from tinygpt.attention import MultiHeadAttention
from tinygpt.ffn import SwiGLU
from tinygpt.norm import RMSNorm


class TransformerBlock(nn.Module):
    """A single pre-norm transformer block (LLaMA / Mistral / Qwen style).

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Key design choices:
        - Pre-norm: normalize BEFORE each sublayer, not after.
        - Two independent RMSNorm layers (separate learned gains).
        - RoPE and causal masking are handled inside MultiHeadAttention.
        - No dropout (LLaMA-style).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 1024,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.attn = MultiHeadAttention(d_model, n_heads, max_seq_len)
        self.ffn_norm = RMSNorm(d_model, eps=eps)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
