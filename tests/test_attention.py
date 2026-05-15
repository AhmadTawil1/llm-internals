import pytest
import torch
import torch.nn.functional as F

from tinygpt.attention import MultiHeadAttention, SingleHeadAttention, scaled_dot_product_attention
from tinygpt.config import TinyGPTConfig

# ── SingleHeadAttention ───────────────────────────────────────────────────────


def test_sha_output_shape() -> None:
    B, T, D = 2, 8, 32
    sha = SingleHeadAttention(TinyGPTConfig(d_model=D))
    x = torch.randn(B, T, D)
    out = sha(x)
    assert out.shape == (B, T, D)


def test_sha_is_causal() -> None:
    """Mutating token t must not affect outputs at positions < t."""
    T, D = 8, 32
    sha = SingleHeadAttention(TinyGPTConfig(d_model=D)).eval()
    x1 = torch.randn(1, T, D)
    x2 = x1.clone()
    x2[:, 4, :] = torch.randn(1, D)
    with torch.no_grad():
        y1 = sha(x1)
        y2 = sha(x2)
    torch.testing.assert_close(y1[:, :4, :], y2[:, :4, :])


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
    mha = MultiHeadAttention(TinyGPTConfig(d_model=D, n_heads=H, max_seq_len=32))
    x = torch.randn(B, T, D)
    out = mha(x)
    assert out.shape == (B, T, D)


@pytest.mark.parametrize("n_heads", [1, 4, 8])
def test_mha_output_shape_fixed_input(n_heads: int) -> None:
    """Output is exactly (B, T, C) regardless of head count."""
    B, T, C = 2, 8, 64
    mha = MultiHeadAttention(TinyGPTConfig(d_model=C, n_heads=n_heads))
    x = torch.randn(B, T, C)
    out = mha(x)
    assert out.shape == (B, T, C)


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


@pytest.mark.parametrize("t", [1, 4, 7])
def test_causal_mask_correctness(t: int) -> None:
    """Mutating exactly token t must leave every output at position < t bit-identical."""
    torch.manual_seed(0)
    T, D = 8, 32
    mha = MultiHeadAttention(TinyGPTConfig(d_model=D, n_heads=4, max_seq_len=16))
    mha.eval()

    x1 = torch.randn(1, T, D)
    x2 = x1.clone()
    x2[:, t, :] = torch.randn(1, D)  # mutate only position t

    with torch.no_grad():
        y1 = mha(x1)
        y2 = mha(x2)

    torch.testing.assert_close(y1[:, :t, :], y2[:, :t, :])


def test_causal_information_does_not_leak():
    """
    Token i's output must not depend on tokens j > i.
    Test: change the future, observe that output[..., :i+1, :] is unchanged.
    """
    torch.manual_seed(0)
    mha = MultiHeadAttention(TinyGPTConfig(d_model=32, n_heads=4, max_seq_len=16))
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
    mha = MultiHeadAttention(TinyGPTConfig(d_model=32, n_heads=4, max_seq_len=8))
    x = torch.randn(2, 4, 32, requires_grad=True)
    loss = mha(x).sum()
    loss.backward()

    for name, p in mha.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert p.grad.abs().sum() > 0, f"{name} grad is all zeros"


# ---------- divisibility check ----------


def test_d_model_must_divide_heads():
    with pytest.raises(AssertionError):
        MultiHeadAttention(TinyGPTConfig(d_model=33, n_heads=4))


# ── pad masking tests ─────────────────────────────────────────────────────────


def test_pad_columns_are_not_attended_to() -> None:
    """A real token must give zero attention weight to pad columns."""
    torch.manual_seed(0)
    attn = MultiHeadAttention(TinyGPTConfig(d_model=64, n_heads=4))
    B, T = 2, 8
    x = torch.randn(B, T, 64)
    mask = torch.ones(B, T)
    mask[1, -3:] = 0

    _, attn_weights = attn(x, attention_mask=mask, return_weights=True)

    real_queries = attn_weights[1, :, :5, :]  # heads, real_q_positions, all_k
    pad_columns = real_queries[..., -3:]
    assert torch.allclose(pad_columns, torch.zeros_like(pad_columns), atol=1e-6), (
        "Real queries are leaking attention into pad columns"
    )


def test_pad_rows_produce_no_nans_downstream() -> None:
    """Pad-row NaN handling: forward pass on a fully-padded position must not poison the batch."""
    torch.manual_seed(0)
    attn = MultiHeadAttention(TinyGPTConfig(d_model=64, n_heads=4))
    B, T = 2, 8
    x = torch.randn(B, T, 64)
    mask = torch.ones(B, T)
    mask[0, 4:] = 0

    out = attn(x, attention_mask=mask)
    assert torch.isfinite(out).all(), "NaNs or Infs in attention output"
