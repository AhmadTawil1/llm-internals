import torch
import torch.nn.functional as F

from tinygpt.ffn import SwiGLU

# ── Shape tests ──────────────────────────────────────────────────────────────


class TestSwiGLUShape:
    def test_output_shape(self) -> None:
        """Output shape must equal input shape: (B, T, d_model)."""
        B, T, d_model, d_ff = 2, 8, 64, 128
        ffn = SwiGLU(d_model, d_ff)
        x = torch.randn(B, T, d_model)
        out = ffn(x)
        assert out.shape == (B, T, d_model)

    def test_different_d_ff(self) -> None:
        """d_ff can be anything — it's an internal dimension."""
        ffn = SwiGLU(d_model=32, d_ff=256)
        x = torch.randn(1, 4, 32)
        assert ffn(x).shape == (1, 4, 32)


# ── Parameter structure ──────────────────────────────────────────────────────


class TestSwiGLUParams:
    def test_three_projections(self) -> None:
        """SwiGLU must have exactly gate_proj, up_proj, down_proj."""
        ffn = SwiGLU(d_model=64, d_ff=128)
        names = {name for name, _ in ffn.named_parameters()}
        assert names == {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}

    def test_no_bias(self) -> None:
        """All three projections should be bias-free."""
        ffn = SwiGLU(d_model=64, d_ff=128)
        assert ffn.gate_proj.bias is None
        assert ffn.up_proj.bias is None
        assert ffn.down_proj.bias is None

    def test_param_count(self) -> None:
        """SwiGLU has 3 × d_model × d_ff parameters (no biases)."""
        d_model, d_ff = 64, 128
        ffn = SwiGLU(d_model, d_ff)
        total = sum(p.numel() for p in ffn.parameters())
        assert total == 3 * d_model * d_ff


# ── Numerical correctness ────────────────────────────────────────────────────


class TestSwiGLUNumerics:
    def test_manual_forward(self) -> None:
        """Verify forward matches the explicit formula:
        down_proj( silu(gate_proj(x)) * up_proj(x) )
        """
        torch.manual_seed(0)
        d_model, d_ff = 16, 32
        ffn = SwiGLU(d_model, d_ff)
        x = torch.randn(1, 1, d_model)

        # Compute manually.
        gate = F.silu(ffn.gate_proj(x))
        up = ffn.up_proj(x)
        expected = ffn.down_proj(gate * up)

        out = ffn(x)
        torch.testing.assert_close(out, expected)


# ── Gradient flow ─────────────────────────────────────────────────────────────


class TestSwiGLUGradients:
    def test_gradients_flow(self) -> None:
        """Backprop produces non-zero gradients on all three projections."""
        ffn = SwiGLU(d_model=32, d_ff=64)
        x = torch.randn(2, 4, 32, requires_grad=True)
        loss = ffn(x).sum()
        loss.backward()

        for name, p in ffn.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert p.grad.abs().sum() > 0, f"{name} grad is all zeros"

    def test_input_grad_flows(self) -> None:
        """Gradient also flows back to the input tensor."""
        ffn = SwiGLU(d_model=32, d_ff=64)
        x = torch.randn(1, 4, 32, requires_grad=True)
        loss = ffn(x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
