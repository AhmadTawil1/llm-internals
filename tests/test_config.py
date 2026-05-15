import pytest

from tinygpt.config import TinyGPTConfig


def test_head_dim_derived() -> None:
    cfg = TinyGPTConfig(d_model=512, n_heads=8)
    assert cfg.head_dim == 64


def test_head_dim_not_a_constructor_arg() -> None:
    with pytest.raises(TypeError):
        TinyGPTConfig(d_model=512, n_heads=8, head_dim=64)  # type: ignore[call-arg]


def test_indivisible_d_model_raises() -> None:
    with pytest.raises(AssertionError):
        TinyGPTConfig(d_model=33, n_heads=4)


def test_defaults() -> None:
    cfg = TinyGPTConfig()
    assert cfg.d_model == 512
    assert cfg.n_heads == 8
    assert cfg.n_layers == 12
    assert cfg.norm_eps == 1e-5
    assert cfg.rope_base == 10000.0
    assert cfg.init_std == 0.02
    assert cfg.dropout == 0.0
