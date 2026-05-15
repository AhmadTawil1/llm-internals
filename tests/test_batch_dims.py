import pytest
import torch

from tinygpt.block import TransformerBlock


def test_forward_works_at_batch_size_one() -> None:
    """Inference time will call this with B=1; it must not error or squeeze a dim."""
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
    x = torch.randn(1, 10, 64)
    out = block(x)
    assert out.shape == (1, 10, 64), f"Expected (1, 10, 64), got {out.shape}"


def test_batch_invariance() -> None:
    """Output for sequence k is independent of what other sequences are in the batch."""
    torch.manual_seed(0)
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256).eval()
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
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
    x = torch.randn(b, t, 64)
    out = block(x)
    assert out.shape == (b, t, 64)


def test_seq_len_one() -> None:
    """Single-token forward is the inference step path. Must not crash."""
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
    x = torch.randn(2, 1, 64)
    out = block(x)
    assert out.shape == (2, 1, 64)
    assert torch.isfinite(out).all()
