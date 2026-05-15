import pytest
import torch

from tinygpt.block import TransformerBlock
from tinygpt.config import TinyGPTConfig


def test_block_output_shape() -> None:
    block = TransformerBlock(TinyGPTConfig(d_model=64, n_heads=4, d_ff=128, max_seq_len=32))
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == x.shape


def test_block_is_identity_when_sublayers_zeroed() -> None:
    block = TransformerBlock(TinyGPTConfig(d_model=64, n_heads=4, d_ff=128, max_seq_len=32))
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.W_o.weight.zero_()
    x = torch.randn(2, 16, 64)
    out = block(x)
    torch.testing.assert_close(out, x)


def test_block_gradients_flow() -> None:
    block = TransformerBlock(TinyGPTConfig(d_model=64, n_heads=4, d_ff=128, max_seq_len=32))
    x = torch.randn(2, 16, 64, requires_grad=True)
    loss = block(x).pow(2).mean()
    loss.backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert p.grad.abs().sum() > 0, f"zero grad for {name}"


def test_block_is_causal() -> None:
    block = TransformerBlock(TinyGPTConfig(d_model=64, n_heads=4, d_ff=128, max_seq_len=32)).eval()
    x1 = torch.randn(1, 16, 64)
    x2 = x1.clone()
    x2[:, 8:, :] += torch.randn(1, 8, 64)
    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)
    torch.testing.assert_close(out1[:, :8, :], out2[:, :8, :])


# ── batch dimension tests ─────────────────────────────────────────────────────


def test_forward_works_at_batch_size_one() -> None:
    """Inference time will call this with B=1; it must not error or squeeze a dim."""
    block = TransformerBlock(TinyGPTConfig(d_model=64, n_heads=4, d_ff=256))
    x = torch.randn(1, 10, 64)
    out = block(x)
    assert out.shape == (1, 10, 64), f"Expected (1, 10, 64), got {out.shape}"


def test_batch_invariance() -> None:
    """Output for sequence k is independent of what other sequences are in the batch."""
    torch.manual_seed(0)
    block = TransformerBlock(TinyGPTConfig(d_model=64, n_heads=4, d_ff=256)).eval()
    x_solo = torch.randn(1, 10, 64)
    x_batch = torch.cat([x_solo, torch.randn(3, 10, 64)], dim=0)  # (4, 10, 64)

    with torch.no_grad():
        out_solo = block(x_solo)
        out_batch = block(x_batch)

    assert torch.allclose(out_solo[0], out_batch[0], atol=1e-5), (
        "Output depends on what other sequences are in the batch — leakage!"
    )


@pytest.mark.parametrize("b,t", [(1, 1), (1, 32), (4, 16), (8, 128), (16, 64)])
def test_shapes_across_batch_and_seq(b: int, t: int) -> None:
    block = TransformerBlock(TinyGPTConfig(d_model=64, n_heads=4, d_ff=256))
    x = torch.randn(b, t, 64)
    out = block(x)
    assert out.shape == (b, t, 64)


def test_seq_len_one() -> None:
    """Single-token forward is the inference step path. Must not crash."""
    block = TransformerBlock(TinyGPTConfig(d_model=64, n_heads=4, d_ff=256))
    x = torch.randn(2, 1, 64)
    out = block(x)
    assert out.shape == (2, 1, 64)
    assert torch.isfinite(out).all()


# ── gradient tests ────────────────────────────────────────────────────────────


def test_every_parameter_gets_a_gradient() -> None:
    """If a parameter has grad=None or grad=0 after backward, it's a dead branch."""
    torch.manual_seed(0)
    block = TransformerBlock(TinyGPTConfig(d_model=64, n_heads=4, d_ff=256))
    x = torch.randn(2, 8, 64, requires_grad=False)
    out = block(x)
    loss = out.sum()
    loss.backward()

    dead = []
    for name, p in block.named_parameters():
        if p.grad is None:
            dead.append((name, "grad is None"))
        elif torch.allclose(p.grad, torch.zeros_like(p.grad)):
            dead.append((name, "grad is all zeros"))
    assert not dead, f"Dead parameters: {dead}"


def test_gradients_are_finite() -> None:
    """NaN or Inf gradients silently destroy training."""
    torch.manual_seed(0)
    block = TransformerBlock(TinyGPTConfig(d_model=64, n_heads=4, d_ff=256))
    x = torch.randn(2, 8, 64)
    out = block(x)
    loss = out.sum()
    loss.backward()

    for name, p in block.named_parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"


def test_gradients_finite_with_padding() -> None:
    """The pad-row-NaN-to-zero trick must produce a valid gradient too."""
    torch.manual_seed(0)
    block = TransformerBlock(TinyGPTConfig(d_model=64, n_heads=4, d_ff=256))
    B, T = 2, 8
    x = torch.randn(B, T, 64)
    mask = torch.ones(B, T)
    mask[0, 5:] = 0  # padding

    out = block(x, attention_mask=mask)
    loss = (out * mask[..., None]).sum()
    loss.backward()

    for name, p in block.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name} with padding"


def test_gradient_magnitudes_are_reasonable_at_init() -> None:
    """Gradient magnitude sanity check at initialization."""
    torch.manual_seed(0)
    block = TransformerBlock(TinyGPTConfig(d_model=64, n_heads=4, d_ff=256))
    x = torch.randn(4, 16, 64)
    loss = block(x).pow(2).mean()
    loss.backward()

    for name, p in block.named_parameters():
        assert p.grad is not None
        gn = p.grad.norm().item()
        assert 1e-6 < gn < 1e3, f"{name}: grad norm {gn:.2e} — bad init?"
