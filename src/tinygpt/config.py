from dataclasses import dataclass, field


@dataclass
class TinyGPTConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    max_seq_len: int = 1024
    vocab_size: int = 32000
    norm_eps: float = 1e-5
    rope_base: float = 10000.0
    init_std: float = 0.02
    dropout: float = 0.0
    # Derived — do not pass to the constructor.
    head_dim: int = field(init=False)

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, (
            f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
        )
        self.head_dim = self.d_model // self.n_heads
