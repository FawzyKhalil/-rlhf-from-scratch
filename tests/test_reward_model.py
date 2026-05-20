"""Unit tests for RewardModel, bradley_terry_loss, and reward_accuracy."""

import torch
import pytest

from src.reward_model import RewardModel, bradley_terry_loss, reward_accuracy


# ---------------------------------------------------------------------------
# Model shape / correctness
# ---------------------------------------------------------------------------

def test_output_shape(tiny_config, device):
    model = RewardModel(tiny_config).to(device)
    B, T = 3, 20
    ids = torch.randint(0, 1000, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.long, device=device)
    scores = model(ids, mask)
    assert scores.shape == (B,), f"Expected ({B},), got {scores.shape}"


def test_output_is_finite(tiny_config, device):
    model = RewardModel(tiny_config).to(device)
    ids = torch.randint(0, 1000, (4, 32), device=device)
    mask = torch.ones(4, 32, dtype=torch.long, device=device)
    scores = model(ids, mask)
    assert torch.isfinite(scores).all(), "Reward scores contain NaN or Inf"


def test_last_nonpadding_token(tiny_config, device):
    """
    Padding should not affect the score of the real part of the sequence.

    We construct two inputs with identical real tokens but different padding
    lengths and verify the scores are equal.
    """
    model = RewardModel(tiny_config).eval().to(device)
    real_len = 10
    real_ids = torch.randint(1, 1000, (1, real_len), device=device)

    # Version A: no padding
    ids_a = real_ids
    mask_a = torch.ones(1, real_len, dtype=torch.long, device=device)

    # Version B: 10 extra pad tokens appended
    pad = torch.zeros(1, 10, dtype=torch.long, device=device)
    ids_b = torch.cat([real_ids, pad], dim=1)
    mask_b = torch.cat([torch.ones(1, real_len), torch.zeros(1, 10)], dim=1).long().to(device)

    with torch.no_grad():
        score_a = model(ids_a, mask_a)
        score_b = model(ids_b, mask_b)

    assert torch.allclose(score_a, score_b, atol=1e-5), (
        f"Padding changed the score: {score_a.item():.6f} vs {score_b.item():.6f}"
    )


def test_no_attention_mask(tiny_config, device):
    """Model should still return a valid scalar when attention_mask is None."""
    model = RewardModel(tiny_config).to(device)
    ids = torch.randint(0, 1000, (2, 16), device=device)
    scores = model(ids, attention_mask=None)
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def test_bradley_terry_loss_decreases_on_correct_ranking():
    r_w = torch.tensor([2.0, 3.0, 1.5])
    r_l = torch.tensor([1.0, 1.0, 0.5])
    loss = bradley_terry_loss(r_w, r_l)
    # Loss should be < log(2) ≈ 0.693 (the random-chance baseline)
    assert 0.0 < loss.item() < 0.693, f"Unexpected loss value: {loss.item()}"


def test_bradley_terry_loss_at_random_chance():
    torch.manual_seed(0)
    r_w = torch.zeros(1000)
    r_l = torch.zeros(1000)
    loss = bradley_terry_loss(r_w, r_l)
    # When r_w == r_l: σ(0) = 0.5, so loss = -log(0.5) = log(2) ≈ 0.693
    assert abs(loss.item() - 0.693) < 0.01, f"Expected ~0.693, got {loss.item()}"


def test_bradley_terry_loss_gradient_flows(tiny_config, device):
    model = RewardModel(tiny_config).to(device)
    ids_w = torch.randint(0, 1000, (2, 16), device=device)
    ids_l = torch.randint(0, 1000, (2, 16), device=device)
    mask = torch.ones(2, 16, dtype=torch.long, device=device)

    r_w = model(ids_w, mask)
    r_l = model(ids_l, mask)
    loss = bradley_terry_loss(r_w, r_l)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed"
    assert all(torch.isfinite(g).all() for g in grads), "Non-finite gradients"


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def test_reward_accuracy_perfect():
    r_w = torch.tensor([2.0, 3.0, 5.0])
    r_l = torch.tensor([1.0, 1.0, 0.5])
    assert reward_accuracy(r_w, r_l) == 1.0


def test_reward_accuracy_zero():
    r_w = torch.tensor([0.0, -1.0])
    r_l = torch.tensor([1.0, 2.0])
    assert reward_accuracy(r_w, r_l) == 0.0


def test_reward_accuracy_near_random():
    torch.manual_seed(42)
    r_w = torch.randn(10_000)
    r_l = torch.randn(10_000)
    acc = reward_accuracy(r_w, r_l)
    assert 0.45 < acc < 0.55, f"Expected ~0.5 for random scores, got {acc:.3f}"
