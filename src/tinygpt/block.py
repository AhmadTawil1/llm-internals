import math

import torch
import torch.nn as nn

from tinygpt.attention import MultiHeadAttention
from tinygpt.config import TinyGPTConfig
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
        config: TinyGPTConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

        self.apply(self._init_weights)

        # Residual projections feed directly into the residual stream and
        # accumulate across n_layers. Scale their std down so the stream
        # variance stays near 1 at init (GPT-2 paper §2.3).
        residual_std = config.init_std / math.sqrt(2 * config.n_layers)
        nn.init.normal_(self.attn.W_o.weight, mean=0.0, std=residual_std)
        nn.init.normal_(self.ffn.down_proj.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
