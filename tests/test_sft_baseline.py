"""Unit tests for SFTModel and SFTTrainer."""

import torch
import pytest

from src.sft_baseline import SFTModel, SFTTrainer


def test_forward_no_labels(tiny_config, device):
    model = SFTModel(model_name=tiny_config, use_lora=False).to(device)
    ids = torch.randint(0, 1000, (2, 16), device=device)
    mask = torch.ones(2, 16, dtype=torch.long, device=device)
    out = model(ids, mask)
    # Without labels, HuggingFace returns logits only; loss is None
    assert out.logits.shape == (2, 16, tiny_config.vocab_size)
    assert out.loss is None


def test_forward_with_labels(tiny_config, device):
    model = SFTModel(model_name=tiny_config, use_lora=False).to(device)
    ids = torch.randint(0, 1000, (2, 16), device=device)
    mask = torch.ones(2, 16, dtype=torch.long, device=device)
    labels = ids.clone()
    out = model(ids, mask, labels=labels)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_lora_reduces_trainable_parameters(tiny_config):
    """LoRA should dramatically reduce the number of trainable parameters."""
    full_model = SFTModel(model_name=tiny_config, use_lora=False)
    lora_model = SFTModel(model_name=tiny_config, use_lora=True, lora_r=4)

    full_trainable = full_model.trainable_parameters()
    lora_trainable = lora_model.trainable_parameters()

    assert lora_trainable < full_trainable, (
        f"LoRA should have fewer trainable params: {lora_trainable} vs {full_trainable}"
    )


def test_trainer_step_reduces_loss(tiny_config, device):
    model = SFTModel(model_name=tiny_config, use_lora=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = SFTTrainer(model, optimizer, device)

    ids = torch.randint(0, 1000, (2, 16), device=device)
    mask = torch.ones(2, 16, dtype=torch.long, device=device)
    labels = ids.clone()

    losses = [trainer.train_step(ids, mask, labels) for _ in range(5)]
    # Loss should generally trend down over 5 steps on a fixed batch
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_total_vs_trainable(tiny_config):
    model = SFTModel(model_name=tiny_config, use_lora=False)
    assert model.total_parameters() >= model.trainable_parameters()
