import torch

from tinygpt.attention import MultiHeadAttention


def test_pad_columns_are_not_attended_to() -> None:
    """A real token must give zero attention weight to pad columns."""
    torch.manual_seed(0)
    attn = MultiHeadAttention(d_model=64, num_heads=4)
    B, T = 2, 8
    x = torch.randn(B, T, 64)
    # Sequence 0: all real. Sequence 1: last 3 positions are pad.
    mask = torch.ones(B, T)
    mask[1, -3:] = 0

    out, attn_weights = attn(x, attention_mask=mask, return_weights=True)

    # For sequence 1, no real query should attend to the pad columns (last 3)
    real_queries = attn_weights[1, :, :5, :]  # heads, real_q_positions, all_k
    pad_columns = real_queries[..., -3:]
    assert torch.allclose(pad_columns, torch.zeros_like(pad_columns), atol=1e-6), (
        "Real queries are leaking attention into pad columns"
    )


def test_pad_rows_produce_no_nans_downstream() -> None:
    """Pad-row NaN handling: forward pass on a fully-padded position must not poison the batch."""
    torch.manual_seed(0)
    attn = MultiHeadAttention(d_model=64, num_heads=4)
    B, T = 2, 8
    x = torch.randn(B, T, 64)
    mask = torch.ones(B, T)
    mask[0, 4:] = 0  # second half of seq 0 is padding

    out = attn(x, attention_mask=mask)
    assert torch.isfinite(out).all(), "NaNs or Infs in attention output"
