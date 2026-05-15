import math

import pytest
import torch

from tinygpt.norm import RMSNorm

# ── Sanity check from the exercise prompt ────────────────────────────────────


class TestRMSNormBasic:
    """Hand-computed example: x = [1, 2, 3, 4], gain = ones."""

    def test_known_values(self) -> None:
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        norm = RMSNorm(4)
        out = norm(x)

        # rms = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) ≈ 2.7386
        rms = math.sqrt(7.5)
        expected = torch.tensor([[1 / rms, 2 / rms, 3 / rms, 4 / rms]])

        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_hand_computed_reference(self) -> None:
        """Exact formula check with non-trivial gain and explicit eps.

        x    = [3, -4, 0, 5]
        gain = [2, 1, 0.5, 3]
        eps  = 1e-6

        mean_sq = (9 + 16 + 0 + 25) / 4 = 12.5
        rms     = sqrt(12.5 + 1e-6)
        output  = (x / rms) * gain
        """
        eps = 1e-6
        x = torch.tensor([[3.0, -4.0, 0.0, 5.0]])
        norm = RMSNorm(4, eps=eps)
        with torch.no_grad():
            norm.gain.copy_(torch.tensor([2.0, 1.0, 0.5, 3.0]))

        out = norm(x)

        rms = math.sqrt(12.5 + eps)
        expected = torch.tensor(
            [
                [
                    3.0 / rms * 2.0,
                    -4.0 / rms * 1.0,
                    0.0 / rms * 0.5,
                    5.0 / rms * 3.0,
                ]
            ]
        )
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)

    def test_output_shape(self) -> None:
        """RMSNorm should not change the shape of its input."""
        x = torch.randn(2, 8, 64)  # (B, T, D)
        norm = RMSNorm(64)
        out = norm(x)
        assert out.shape == x.shape

    def test_gain_is_learnable(self) -> None:
        """The gain parameter should appear in the module's parameters."""
        norm = RMSNorm(32)
        param_names = [name for name, _ in norm.named_parameters()]
        assert "gain" in param_names


# ── Float32 up-casting ────────────────────────────────────────────────────────


class TestRMSNormDtype:
    """Verify that bf16 inputs produce bf16 outputs (computed in f32 internally)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for bf16")
    def test_bf16_output_dtype(self) -> None:
        norm = RMSNorm(16).cuda().bfloat16()
        x = torch.randn(1, 4, 16, device="cuda", dtype=torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16

    def test_fp16_output_dtype(self) -> None:
        norm = RMSNorm(16).half()
        x = torch.randn(1, 4, 16, dtype=torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16


# ── Unit-norm property ────────────────────────────────────────────────────────


class TestRMSNormProperty:
    """After RMSNorm (with gain=1), the RMS of the output should be ≈ 1."""

    def test_output_rms_is_one(self) -> None:
        torch.manual_seed(42)
        norm = RMSNorm(128)
        x = torch.randn(4, 16, 128)
        out = norm(x)

        # RMS of each token's embedding should be close to 1.
        rms_out = out.pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(
            rms_out,
            torch.ones_like(rms_out),
            atol=1e-5,
            rtol=1e-5,
        )
