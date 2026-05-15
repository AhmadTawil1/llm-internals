import torch

from tinygpt.ffn import SwiGLU


def test_ffn_shape() -> None:
    ffn = SwiGLU(d_model=64, d_ff=128)
    x = torch.randn(2, 16, 64)
    out = ffn(x)
    assert out.shape == (2, 16, 64)


def test_ffn_per_position_independence() -> None:
    ffn = SwiGLU(d_model=64, d_ff=128)
    x1 = torch.randn(1, 16, 64)
    x2 = x1.clone()

    # Replace tokens at position 5
    x2[:, 5, :] = torch.randn(1, 64)

    out1 = ffn(x1)
    out2 = ffn(x2)

    # Other positions must remain entirely identical
    torch.testing.assert_close(out1[:, :5, :], out2[:, :5, :])
    torch.testing.assert_close(out1[:, 6:, :], out2[:, 6:, :])


def test_ffn_param_count() -> None:
    d_model, d_ff = 64, 128
    ffn = SwiGLU(d_model, d_ff)

    total_params = sum(p.numel() for p in ffn.parameters())
    assert total_params == 3 * d_model * d_ff
