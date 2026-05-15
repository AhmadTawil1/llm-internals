import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic() -> None:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
