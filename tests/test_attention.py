import pytest
import torch
import torch.nn.functional as F

from tinygpt.attention import MultiHeadAttention, scaled_dot_product_attention

# ---------- shape tests ----------


def test_sdpa_output_shape():
    """Output of attention has the same shape as V."""
    B, H, T, d_k, d_v = 2, 4, 8, 16, 16
    q = torch.randn(B, H, T, d_k)
    k = torch.randn(B, H, T, d_k)
    v = torch.randn(B, H, T, d_v)
    out, attn = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, T, d_v)
    assert attn.shape == (B, H, T, T)


def test_attn_weights_sum_to_one():
    """Each row of the attention matrix is a probability distribution."""
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    _, attn = scaled_dot_product_attention(q, k, v)
    sums = attn.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-5)


def test_mha_output_shape():
    B, T, D, H = 2, 16, 64, 8
    mha = MultiHeadAttention(d_model=D, num_heads=H, max_seq_len=32)
    x = torch.randn(B, T, D)
    out = mha(x)
    assert out.shape == (B, T, D)


# ---------- causal mask tests ----------


def test_causal_mask_blocks_future():
    """With a causal mask, attention weights on future positions must be exactly zero."""
    B, H, T, d = 1, 1, 6, 4
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    v = torch.randn(B, H, T, d)

    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    _, attn = scaled_dot_product_attention(q, k, v, mask)

    # Upper triangle (future positions, j > i) must be exactly 0.
    upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
    assert (attn[0, 0][upper] == 0).all(), "future positions leaked into attention"


def test_causal_information_does_not_leak():
    """
    Token i's output must not depend on tokens j > i.
    Test: change the future, observe that output[..., :i+1, :] is unchanged.
    """
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=32, num_heads=4, max_seq_len=16)
    mha.eval()  # disable dropout

    x1 = torch.randn(1, 8, 32)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn(1, 3, 32)  # change tokens 5,6,7

    with torch.no_grad():
        y1 = mha(x1)
        y2 = mha(x2)

    # Outputs at positions 0..4 must be identical.
    torch.testing.assert_close(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-5)


# ---------- numerical correctness tests ----------


def test_matches_pytorch_reference():
    """Our SDPA matches torch.nn.functional.scaled_dot_product_attention (no mask case)."""
    B, H, T, d = 2, 4, 8, 16
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    v = torch.randn(B, H, T, d)
    ours, _ = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)


def test_matches_pytorch_reference_causal():
    """Same, with causal mask."""
    B, H, T, d = 2, 4, 8, 16
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    v = torch.randn(B, H, T, d)

    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    ours, _ = scaled_dot_product_attention(q, k, v, mask)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)


# ---------- gradient flow ----------


def test_gradients_flow_through_mha():
    """Backprop produces non-zero gradients on every learnable weight."""
    mha = MultiHeadAttention(d_model=32, num_heads=4, max_seq_len=8)
    x = torch.randn(2, 4, 32, requires_grad=True)
    loss = mha(x).sum()
    loss.backward()

    for name, p in mha.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert p.grad.abs().sum() > 0, f"{name} grad is all zeros"


# ---------- divisibility check ----------


def test_d_model_must_divide_heads():
    with pytest.raises(AssertionError):
        MultiHeadAttention(d_model=33, num_heads=4)
