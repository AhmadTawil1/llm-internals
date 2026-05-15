import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020).

    Used by LLaMA, Mistral, Qwen, and most modern LLMs in place of
    the original transformer's ReLU/GELU MLP.  SwiGLU has three linear
    projections instead of two and consistently outperforms vanilla
    activations at the same parameter count.

    Math:
        FFN(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )

    where silu(x) = x · σ(x)  (the Swish activation).

    Sizing rule of thumb:
        d_ff ≈ (8/3) × d_model, rounded to a multiple of 64 or 128.
        LLaMA-7B: d_model=4096, d_ff=11008.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        # Three projections, all bias-free (modern default).
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        # gate and up are both (B, T, d_ff)
        # element-wise multiply after silu gating, then project back down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))  # type: ignore[no-any-return]
