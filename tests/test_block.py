import torch

from tinygpt.block import TransformerBlock


def test_block_output_shape() -> None:
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=128, max_seq_len=32)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == x.shape


def test_block_is_identity_when_sublayers_zeroed() -> None:
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=128, max_seq_len=32)
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.W_o.weight.zero_()  # In our implementation, it's W_o
    x = torch.randn(2, 16, 64)
    out = block(x)
    torch.testing.assert_close(out, x)


def test_block_gradients_flow() -> None:
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=128, max_seq_len=32)
    x = torch.randn(2, 16, 64, requires_grad=True)
    loss = block(x).pow(2).mean()
    loss.backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert p.grad.abs().sum() > 0, f"zero grad for {name}"


def test_block_is_causal() -> None:
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=128, max_seq_len=32).eval()
    x1 = torch.randn(1, 16, 64)
    x2 = x1.clone()
    x2[:, 8:, :] += torch.randn(1, 8, 64)  # mutate positions 8..15
    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)
    torch.testing.assert_close(out1[:, :8, :], out2[:, :8, :])
