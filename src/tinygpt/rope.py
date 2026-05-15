import torch


def precompute_freqs_cis(
    dim: int,
    end: int,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin tables for Rotary Position Embeddings.

    Args:
        dim:  head dimension (must be even).
        end:  maximum sequence length to precompute for.
        base: frequency base (10 000 in the original paper).

    Returns:
        cos, sin — each of shape (end, dim/2), float32.

    How it works:
        1. Build the frequency vector θ for each pair index i:
               θᵢ = base^(−2i / d)      i ∈ {0, 1, …, d/2 − 1}
           This gives a geometric progression from 1.0 down to base^(−1).
           Low-index pairs rotate fast (high frequency), high-index pairs
           rotate slowly — so nearby tokens differ most on the first dims
           and long-range position information lives in the last dims.

        2. Build the position vector m = [0, 1, 2, …, end−1].

        3. Outer-product m ⊗ θ gives the rotation angle for every
           (position, pair) combination → shape (end, d/2).

        4. Take cos and sin of those angles.
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"

    # θᵢ = base^(−2i/d)  →  shape (d/2,)
    i = torch.arange(0, dim, 2, dtype=torch.float32)  # [0, 2, 4, …, d−2]
    freqs = base ** (-i / dim)  # equivalent to 1 / base^(2i/d)

    # m = [0, 1, 2, …, end−1]  →  shape (end,)
    positions = torch.arange(end, dtype=torch.float32)

    # Outer product: angles[m, i] = m · θᵢ  →  shape (end, d/2)
    angles = torch.outer(positions, freqs)

    return angles.cos(), angles.sin()


def apply_rotary_emb(
    xq: torch.Tensor,  # (B, T, H, D)
    xk: torch.Tensor,  # (B, T, H, D)
    cos: torch.Tensor,  # (T, D/2)  from precompute_freqs_cis
    sin: torch.Tensor,  # (T, D/2)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Uses the GPT-NeoX / LLaMA half-split convention:
        first half  = dims [0 … D/2−1]
        second half = dims [D/2 … D−1]

    So pair i is (dim i, dim i + D/2).

    The rotation for each pair:
        x'₁ = x₁ · cos(mθ) − x₂ · sin(mθ)
        x'₂ = x₁ · sin(mθ) + x₂ · cos(mθ)

    Implementation detail: compute in float32 to avoid precision loss
    in bf16/fp16, then cast back.
    """
    input_dtype = xq.dtype

    # Upcast for precision.
    xq_f32 = xq.float()
    xk_f32 = xk.float()

    # Reshape cos/sin from (T, D/2) → (1, T, 1, D/2) to broadcast
    # over batch (B) and heads (H).
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        # Split into two halves along the last dim.
        #   x1 = x[..., :D/2]   (the "real" half)
        #   x2 = x[..., D/2:]   (the "imaginary" half)
        d_half = x.shape[-1] // 2
        x1, x2 = x[..., :d_half], x[..., d_half:]

        # Apply the 2D rotation to each pair.
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # Concatenate back to (..., D).
        return torch.cat([out1, out2], dim=-1)

    xq_out = _rotate(xq_f32).to(input_dtype)
    xk_out = _rotate(xk_f32).to(input_dtype)

    return xq_out, xk_out
