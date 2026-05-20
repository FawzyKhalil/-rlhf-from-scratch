"""Shared fixtures for unit tests. Uses tiny GPT-2 configs to avoid weight downloads."""

import pytest
import torch
from transformers import GPT2Config


@pytest.fixture(scope="session")
def tiny_config() -> GPT2Config:
    """Minimal GPT-2 config: fast to instantiate, no pretrained weights needed."""
    return GPT2Config(
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_positions=128,
        n_ctx=128,
        vocab_size=1000,
    )


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")
