import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm skips the mean-centering step and only
    rescales by the root-mean-square statistic.  This makes it ~10-15 %
    cheaper while performing comparably in practice.  LLaMA, Mistral,
    and Gemma all use RMSNorm.

    Math:
        rms(x) = sqrt( (1/d) * Σ xᵢ² )
        RMSNorm(x) = (x / (rms(x) + eps)) * g

    where g ∈ ℝᵈ is a learnable per-dimension gain (initialized to ones).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable per-dimension gain, initialized to ones.
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)  — typically (B, T, D) in a transformer.
        #
        # Implementation detail: compute in float32 even if input is bf16/fp16.
        # Squaring small half-precision values can underflow to zero,
        # silently killing the gradient.
        input_dtype = x.dtype
        x_f32 = x.float()

        # Mean of squares over the last dim.  keepdim=True so the result
        # broadcasts back to (..., d) when we divide.
        mean_sq = x_f32.pow(2).mean(dim=-1, keepdim=True)  # (..., 1)

        # RMS + eps for numerical stability.
        rms = torch.sqrt(mean_sq + self.eps)  # (..., 1)

        # Normalize, then cast back to the original dtype before applying gain.
        x_normed = (x_f32 / rms).to(input_dtype)  # (..., d)

        return x_normed * self.gain  # (..., d)
