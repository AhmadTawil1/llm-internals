import math

import pytest
import torch

from tinygpt.rope import apply_rotary_emb, precompute_freqs_cis

# ── precompute_freqs_cis ─────────────────────────────────────────────────────

class TestPrecomputeFreqsCis:

    def test_output_shapes(self) -> None:
        dim, end = 8, 16
        cos, sin = precompute_freqs_cis(dim, end)
        assert cos.shape == (end, dim // 2)
        assert sin.shape == (end, dim // 2)

    def test_position_zero_is_identity(self) -> None:
        """At position 0 the angle is 0, so cos=1, sin=0 — no rotation."""
        cos, sin = precompute_freqs_cis(dim=8, end=4)
        torch.testing.assert_close(cos[0], torch.ones(4))
        torch.testing.assert_close(sin[0], torch.zeros(4))

    def test_known_angle_position_one(self) -> None:
        """At position 1, angle_i = θ_i = base^(−2i/d).  Check pair 0."""
        dim, base = 8, 10_000.0
        cos, sin = precompute_freqs_cis(dim, end=2, base=base)

        # Pair 0: θ₀ = base^(0/8) = 1.0, so angle = 1·1.0 = 1.0 radian.
        torch.testing.assert_close(cos[1, 0], torch.tensor(math.cos(1.0)), atol=1e-6, rtol=0)
        torch.testing.assert_close(sin[1, 0], torch.tensor(math.sin(1.0)), atol=1e-6, rtol=0)

    def test_odd_dim_raises(self) -> None:
        with pytest.raises(AssertionError, match="even"):
            precompute_freqs_cis(dim=7, end=4)


# ── apply_rotary_emb ─────────────────────────────────────────────────────────

class TestApplyRotaryEmb:

    def test_output_shapes(self) -> None:
        B, T, H, D = 2, 8, 4, 16
        xq = torch.randn(B, T, H, D)
        xk = torch.randn(B, T, H, D)
        cos, sin = precompute_freqs_cis(D, T)
        q_out, k_out = apply_rotary_emb(xq, xk, cos, sin)
        assert q_out.shape == (B, T, H, D)
        assert k_out.shape == (B, T, H, D)

    def test_position_zero_is_identity(self) -> None:
        """At position 0, rotation angle is 0 → output == input."""
        B, H, D = 1, 1, 8
        xq = torch.randn(B, 1, H, D)
        xk = torch.randn(B, 1, H, D)
        cos, sin = precompute_freqs_cis(D, end=1)

        q_out, k_out = apply_rotary_emb(xq, xk, cos, sin)
        torch.testing.assert_close(q_out, xq, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_out, xk, atol=1e-5, rtol=1e-5)

    def test_norm_preservation(self) -> None:
        """Rotation must not change the vector's L2 norm."""
        B, T, H, D = 2, 16, 4, 32
        xq = torch.randn(B, T, H, D)
        xk = torch.randn(B, T, H, D)
        cos, sin = precompute_freqs_cis(D, T)

        q_out, _ = apply_rotary_emb(xq, xk, cos, sin)

        norm_before = xq.norm(dim=-1)
        norm_after = q_out.norm(dim=-1)
        torch.testing.assert_close(norm_before, norm_after, atol=1e-4, rtol=1e-4)

    def test_relative_position_invariance(self) -> None:
        """The dot product q·k should depend only on the RELATIVE position
        (m − n), not the absolute positions m and n.

        Claim:  RoPE(q, m) · RoPE(k, n)  =  f(q, k, m − n)

        We test three (m, n) pairs that all share the same offset n − m = 3:
            (5, 8),  (10, 13),  (20, 23)
        All three must produce the same dot product (within fp32 tolerance).

        Then we test a pair with a DIFFERENT offset (5, 10) → offset 5, and
        assert the dot product is NOT equal to the others.
        """
        torch.manual_seed(0)
        head_dim = 64
        seq_len = 64
        cos, sin = precompute_freqs_cis(head_dim, seq_len)

        q = torch.randn(1, 1, 1, head_dim)
        k = torch.randn(1, 1, 1, head_dim)

        def rotated_dot(m: int, n: int) -> float:
            """Rotate q to position m, k to position n, return their dot product."""
            q_rot, _ = apply_rotary_emb(q, q, cos[m : m + 1], sin[m : m + 1])
            _, k_rot = apply_rotary_emb(k, k, cos[n : n + 1], sin[n : n + 1])
            return (q_rot * k_rot).sum().item()

        # ── Same relative distance (n − m = 3) → same dot product ──
        d1 = rotated_dot(5, 8)
        d2 = rotated_dot(10, 13)
        d3 = rotated_dot(20, 23)
        assert abs(d1 - d2) < 1e-5, f"d1={d1}, d2={d2}"
        assert abs(d1 - d3) < 1e-5, f"d1={d1}, d3={d3}"

        # ── Different relative distance (n − m = 5) → different dot product ──
        d_other = rotated_dot(5, 10)
        assert abs(d1 - d_other) > 1e-3, (
            f"Different offsets should give different dots: d1={d1}, d_other={d_other}"
        )

    def test_hand_computed_rotation(self) -> None:
        """Verify a single 2D rotation by hand.

        With D=2, the half-split gives x1=[x₀], x2=[x₁].
        At position 1 with base=10000, θ₀ = 10000^(0/2) = 1.0.
        angle = 1·1.0 = 1.0 radian.

            x'₀ = x₀·cos(1) − x₁·sin(1)
            x'₁ = x₀·sin(1) + x₁·cos(1)
        """
        D = 2
        cos, sin = precompute_freqs_cis(D, end=2)
        x = torch.tensor([[[[3.0, 4.0]]]])  # (1, 1, 1, 2) — position 1

        c, s = math.cos(1.0), math.sin(1.0)
        expected = torch.tensor([[[[3.0 * c - 4.0 * s, 3.0 * s + 4.0 * c]]]])

        out, _ = apply_rotary_emb(x, x, cos[1:2], sin[1:2])
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


# ── Dtype preservation ────────────────────────────────────────────────────────

class TestRoPEDtype:

    def test_fp16_output_dtype(self) -> None:
        B, T, H, D = 1, 4, 2, 8
        cos, sin = precompute_freqs_cis(D, T)
        xq = torch.randn(B, T, H, D, dtype=torch.float16)
        xk = torch.randn(B, T, H, D, dtype=torch.float16)
        q_out, k_out = apply_rotary_emb(xq, xk, cos, sin)
        assert q_out.dtype == torch.float16
        assert k_out.dtype == torch.float16
