import torch

from tinygpt.block import TransformerBlock


def test_every_parameter_gets_a_gradient() -> None:
    """If a parameter has grad=None or grad=0 after backward, it's a dead branch."""
    torch.manual_seed(0)
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
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
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
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
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
    B, T = 2, 8
    x = torch.randn(B, T, 64)
    mask = torch.ones(B, T)
    mask[0, 5:] = 0  # padding

    out = block(x, attention_mask=mask)
    # Compute loss only over real positions — this is what the real training loop will do
    loss = (out * mask[..., None]).sum()
    loss.backward()

    for name, p in block.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name} with padding"


def test_gradient_magnitudes_are_reasonable_at_init() -> None:
    """Gradient magnitude sanity check at initialization."""
    torch.manual_seed(0)
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
    x = torch.randn(4, 16, 64)
    loss = block(x).pow(2).mean()
    loss.backward()

    for name, p in block.named_parameters():
        assert p.grad is not None
        gn = p.grad.norm().item()
        assert 1e-6 < gn < 1e3, f"{name}: grad norm {gn:.2e} — bad init?"
