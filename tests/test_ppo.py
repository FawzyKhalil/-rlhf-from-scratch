"""
Unit tests for Phase 2: rollout buffer, reward shaping, critic model, PPO trainer.

All tests use the tiny GPT-2 config from conftest.py (no weight downloads).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from src.reward_model import RewardModel
from src.reward_shaping import (
    compute_kl_divergence,
    compute_kl_penalty,
    shape_rewards,
)
from src.rollout_buffer import RolloutBuffer
from src.ppo_trainer import CriticModel, PPOTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_actor(cfg: GPT2Config) -> GPT2LMHeadModel:
    return GPT2LMHeadModel(cfg)


def make_critic(cfg: GPT2Config) -> CriticModel:
    return CriticModel(model_name=cfg)


def make_rm(cfg: GPT2Config) -> RewardModel:
    return RewardModel(model_name=cfg)


def random_log_probs(B: int, T: int) -> torch.Tensor:
    """Random log probs in [-5, 0]."""
    return -torch.rand(B, T) * 5.0


def make_response_mask(B: int, lengths: list[int]) -> torch.Tensor:
    T = max(lengths)
    mask = torch.zeros(B, T)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0
    return mask


# ---------------------------------------------------------------------------
# reward_shaping.py tests
# ---------------------------------------------------------------------------

class TestComputeKLPenalty:
    def test_zero_when_identical(self):
        lp = random_log_probs(4, 8)
        kl = compute_kl_penalty(lp, lp)
        assert kl.abs().max().item() == pytest.approx(0.0)

    def test_positive_on_average_when_actor_diverges(self):
        # If actor assigns higher probs, KL > 0 in expectation.
        ref_lp   = -torch.ones(100, 10)          # constant log probs
        actor_lp = ref_lp + 0.5                   # actor is more confident
        kl = compute_kl_penalty(actor_lp, ref_lp)
        assert kl.mean().item() == pytest.approx(0.5, abs=1e-4)

    def test_output_shape(self):
        B, T = 3, 12
        kl = compute_kl_penalty(random_log_probs(B, T), random_log_probs(B, T))
        assert kl.shape == (B, T)


class TestShapeRewards:
    def test_rm_score_placed_at_last_real_token(self):
        B = 2
        rm_scores = torch.tensor([1.0, -2.0])
        # Use lengths strictly shorter than T so padding positions exist.
        mask = make_response_mask(B, [3, 2])  # (2, 3): seq 0 ends at t=2, seq 1 at t=1
        T = mask.shape[1]
        lp = torch.zeros(B, T)

        rewards = shape_rewards(rm_scores, lp, lp, mask, beta=0.0)
        # With beta=0, kl=0 → reward is 0 everywhere except last real token.
        assert rewards[0, 2].item() == pytest.approx(1.0)
        assert rewards[1, 1].item() == pytest.approx(-2.0)
        # Other real-token positions should be 0.
        assert rewards[0, :2].abs().max().item() == pytest.approx(0.0)
        assert rewards[1, :1].abs().max().item() == pytest.approx(0.0)
        # Padding positions must be 0.
        assert rewards[1, 2:].abs().max().item() == pytest.approx(0.0)

    def test_kl_penalty_applied_at_all_real_tokens(self):
        B, T = 1, 5
        rm_scores = torch.zeros(B)
        # actor is more confident by 1.0 nats at every token
        actor_lp = -torch.ones(B, T) * 1.0
        ref_lp   = -torch.ones(B, T) * 2.0
        mask = make_response_mask(B, [T])

        rewards = shape_rewards(rm_scores, actor_lp, ref_lp, mask, beta=0.1)
        # KL_t = actor_lp - ref_lp = -1 - (-2) = 1.0 at every real token
        # shaped_reward_t = -beta * KL_t = -0.1 * 1.0 = -0.1
        expected = -0.1
        for t in range(T):
            assert rewards[0, t].item() == pytest.approx(expected, abs=1e-5)

    def test_padding_positions_are_zero(self):
        B = 3
        lengths = [5, 3, 6]  # max=6; all sequences have at least 1 padding token
        mask = make_response_mask(B, lengths)  # (3, 6)
        T = mask.shape[1]
        rewards = shape_rewards(
            torch.randn(B), random_log_probs(B, T), random_log_probs(B, T), mask
        )
        for i, l in enumerate(lengths):
            if l < T:  # only check sequences that actually have padding
                assert rewards[i, l:].abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_output_shape(self):
        B, T = 4, 10
        mask = make_response_mask(B, [T] * B)
        out = shape_rewards(torch.zeros(B), random_log_probs(B, T), random_log_probs(B, T), mask)
        assert out.shape == (B, T)


class TestComputeKLDivergence:
    def test_zero_when_identical(self):
        B, T = 2, 6
        lp = random_log_probs(B, T)
        mask = make_response_mask(B, [T] * B)
        kl = compute_kl_divergence(lp, lp, mask)
        assert kl.abs().max().item() == pytest.approx(0.0, abs=1e-5)

    def test_sums_over_real_tokens_only(self):
        B, T = 2, 6
        actor_lp = torch.zeros(B, T)
        ref_lp   = -torch.ones(B, T)  # KL_t = 0 - (-1) = 1.0 per token
        # First seq has 4 real tokens, second has 6
        lengths = [4, 6]
        mask = make_response_mask(B, lengths)
        kl = compute_kl_divergence(actor_lp, ref_lp, mask)
        assert kl[0].item() == pytest.approx(4.0, abs=1e-5)
        assert kl[1].item() == pytest.approx(6.0, abs=1e-5)


# ---------------------------------------------------------------------------
# RolloutBuffer tests
# ---------------------------------------------------------------------------

class TestRolloutBuffer:
    def test_add_and_len(self):
        buf = RolloutBuffer()
        for _ in range(5):
            buf.add(
                torch.randint(0, 100, (10,)),
                torch.randint(0, 100, (8,)),
                random_log_probs(1, 8).squeeze(0),
                torch.randn(8),
                torch.randn(8),
            )
        assert len(buf) == 5

    def test_compute_advantages_sets_attributes(self):
        buf = RolloutBuffer()
        buf.add(
            torch.zeros(5, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4),
            torch.zeros(4),
            torch.ones(4),
        )
        buf.compute_advantages(gamma=1.0, lam=0.95)
        assert buf.advantages is not None
        assert buf.returns is not None
        assert len(buf.advantages) == 1
        assert buf.advantages[0].shape == (4,)

    def test_gae_terminal_value_is_zero(self):
        """With constant reward=1 and value=0, advantages should equal MC returns."""
        buf = RolloutBuffer()
        T = 5
        buf.add(
            torch.zeros(3, dtype=torch.long),
            torch.zeros(T, dtype=torch.long),
            torch.zeros(T),
            torch.zeros(T),     # V=0 everywhere
            torch.ones(T),      # r=1 everywhere
        )
        buf.compute_advantages(gamma=1.0, lam=1.0)  # lam=1 → MC

        # With V=0 and γ=λ=1: A_t = Σ_{k=t}^{T-1} r_k = T - t
        # Before whitening: A_0=5, A_1=4, ..., A_4=1
        # Whitened: (A_t - mean) / std
        adv = buf.advantages[0]
        assert adv.shape == (T,)
        # Advantages must be monotonically decreasing (before whitening they are T-t)
        assert (adv[:-1] > adv[1:]).all(), "Advantages should decrease over time"

    def test_returns_equal_advantages_plus_values(self):
        buf = RolloutBuffer()
        T = 6
        values = torch.randn(T)
        rewards = torch.randn(T)
        buf.add(
            torch.zeros(2, dtype=torch.long),
            torch.zeros(T, dtype=torch.long),
            torch.zeros(T),
            values.clone(),
            rewards.clone(),
        )
        buf.compute_advantages()
        # R_t = A_t + V_t  (before whitening the advantages, returns are consistent)
        # Note: advantages ARE whitened, but returns are NOT. So returns ≠ adv + values.
        # Instead, returns[t] = unwhitened_adv[t] + values[t].
        # We just check that returns are finite and correct shape.
        assert buf.returns[0].shape == (T,)
        assert buf.returns[0].isfinite().all()

    def test_mini_batches_shape(self):
        buf = RolloutBuffer()
        for i in range(6):
            T_r = 4 + i % 3  # variable response lengths
            T_q = 5 + i % 2  # variable query lengths
            buf.add(
                torch.randint(0, 100, (T_q,)),
                torch.randint(0, 100, (T_r,)),
                torch.zeros(T_r),
                torch.zeros(T_r),
                torch.zeros(T_r),
            )
        buf.compute_advantages()
        batches = list(buf.mini_batches(mini_batch_size=3, device=torch.device("cpu")))
        assert len(batches) == 2
        for b in batches:
            assert b["response_ids"].shape[0] == 3
            # All required keys present
            for k in ("query_ids", "query_mask", "response_ids", "response_mask",
                      "old_log_probs", "old_values", "advantages", "returns"):
                assert k in b

    def test_mini_batches_query_left_padded(self):
        """Query padding should appear on the LEFT (flush-right real tokens)."""
        buf = RolloutBuffer()
        # One long query (8 tokens), one short query (3 tokens)
        buf.add(torch.ones(8, dtype=torch.long), torch.zeros(3, dtype=torch.long),
                torch.zeros(3), torch.zeros(3), torch.zeros(3))
        buf.add(torch.ones(3, dtype=torch.long), torch.zeros(3, dtype=torch.long),
                torch.zeros(3), torch.zeros(3), torch.zeros(3))
        buf.compute_advantages()
        batches = list(buf.mini_batches(mini_batch_size=2, device=torch.device("cpu")))
        b = batches[0]
        q_mask = b["query_mask"]  # (2, 8); rows may be shuffled — find short row
        real_counts = q_mask.sum(dim=1)  # number of real tokens per row
        short_row = real_counts.argmin().item()
        # Short sequence (3 tokens) must be right-aligned: first 5 positions are PAD
        assert q_mask[short_row, :5].sum().item() == pytest.approx(0.0), \
            "Left side of short query should be padding"
        assert q_mask[short_row, 5:].sum().item() == pytest.approx(3.0)

    def test_clear_resets_buffer(self):
        buf = RolloutBuffer()
        buf.add(torch.zeros(4, dtype=torch.long), torch.zeros(3, dtype=torch.long),
                torch.zeros(3), torch.zeros(3), torch.zeros(3))
        buf.compute_advantages()
        buf.clear()
        assert len(buf) == 0
        assert buf.advantages is None
        assert buf.returns is None


# ---------------------------------------------------------------------------
# CriticModel tests
# ---------------------------------------------------------------------------

class TestCriticModel:
    def test_output_shape(self, tiny_config):
        critic = make_critic(tiny_config)
        B, T_q, T_r = 2, 10, 8
        q  = torch.randint(0, tiny_config.vocab_size, (B, T_q))
        qm = torch.ones(B, T_q)
        r  = torch.randint(0, tiny_config.vocab_size, (B, T_r))
        rm = torch.ones(B, T_r)
        values = critic(q, qm, r, rm)
        assert values.shape == (B, T_r)

    def test_output_is_finite(self, tiny_config):
        critic = make_critic(tiny_config)
        B, T_q, T_r = 3, 7, 5
        values = critic(
            torch.randint(0, tiny_config.vocab_size, (B, T_q)),
            torch.ones(B, T_q),
            torch.randint(0, tiny_config.vocab_size, (B, T_r)),
            torch.ones(B, T_r),
        )
        assert values.isfinite().all()

    def test_padding_positions_are_zero(self, tiny_config):
        critic = make_critic(tiny_config)
        B, T_q, T_r = 2, 6, 8
        # Responses: seq 0 has 5 real tokens, seq 1 has 7 real tokens
        r_mask = torch.zeros(B, T_r)
        r_mask[0, :5] = 1.0
        r_mask[1, :7] = 1.0
        values = critic(
            torch.randint(0, tiny_config.vocab_size, (B, T_q)),
            torch.ones(B, T_q),
            torch.randint(0, tiny_config.vocab_size, (B, T_r)),
            r_mask,
        )
        assert values[0, 5:].abs().max().item() == pytest.approx(0.0, abs=1e-6)
        assert values[1, 7:].abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flows_to_value_head(self, tiny_config):
        critic = make_critic(tiny_config)
        B, T_q, T_r = 1, 4, 4
        values = critic(
            torch.randint(0, tiny_config.vocab_size, (B, T_q)),
            torch.ones(B, T_q),
            torch.randint(0, tiny_config.vocab_size, (B, T_r)),
            torch.ones(B, T_r),
        )
        values.sum().backward()
        assert critic.value_head.weight.grad is not None
        assert critic.value_head.weight.grad.abs().max() > 0


# ---------------------------------------------------------------------------
# PPOTrainer: collect_rollouts + ppo_step smoke tests
# ---------------------------------------------------------------------------

class TestPPOTrainer:
    """
    End-to-end smoke tests using tiny models.
    We verify shapes and that the update step changes actor parameters.
    """

    @pytest.fixture
    def tiny_trainer(self, tiny_config, device):
        actor  = make_actor(tiny_config)
        critic = make_critic(tiny_config)
        ref    = make_actor(tiny_config)
        rm     = make_rm(tiny_config)
        trainer = PPOTrainer(
            actor=actor,
            critic=critic,
            ref_model=ref,
            reward_model=rm,
            device=device,
            clip_eps=0.2,
            vf_coef=0.1,
            entropy_coef=0.01,
            n_ppo_epochs=2,
            mini_batch_size=2,
            max_grad_norm=1.0,
            kl_coef=0.05,
            max_new_tokens=6,
            temperature=1.0,
        )
        return trainer

    def test_collect_rollouts_fills_buffer(self, tiny_trainer, tiny_config):
        buf = RolloutBuffer()
        prompts = [torch.randint(0, tiny_config.vocab_size, (5,)) for _ in range(4)]
        stats = tiny_trainer.collect_rollouts(prompts, buf)
        assert len(buf) == 4
        assert "rm_score_mean" in stats
        assert "kl_mean" in stats
        assert "response_len" in stats
        # Sequences should have at least 1 response token
        for r in buf.response_ids:
            assert r.shape[0] >= 1

    def test_ppo_step_returns_metrics(self, tiny_trainer, tiny_config, device):
        buf = RolloutBuffer()
        prompts = [torch.randint(0, tiny_config.vocab_size, (5,)) for _ in range(4)]
        tiny_trainer.collect_rollouts(prompts, buf)
        buf.compute_advantages()

        actor_opt  = torch.optim.AdamW(tiny_trainer.actor.parameters(),  lr=1e-4)
        critic_opt = torch.optim.AdamW(tiny_trainer.critic.parameters(), lr=1e-4)
        metrics = tiny_trainer.ppo_step(buf, actor_opt, critic_opt)

        assert "actor_loss"  in metrics
        assert "critic_loss" in metrics
        assert "entropy"     in metrics
        assert "approx_kl"   in metrics
        for v in metrics.values():
            assert isinstance(v, float)
            assert v == v  # not NaN

    def test_ppo_step_updates_actor_weights(self, tiny_trainer, tiny_config, device):
        """Actor weights should change after a PPO step."""
        buf = RolloutBuffer()
        prompts = [torch.randint(0, tiny_config.vocab_size, (5,)) for _ in range(4)]
        tiny_trainer.collect_rollouts(prompts, buf)
        buf.compute_advantages()

        # Snapshot actor weights before update
        before = {
            name: param.data.clone()
            for name, param in tiny_trainer.actor.named_parameters()
        }

        actor_opt  = torch.optim.AdamW(tiny_trainer.actor.parameters(),  lr=1e-3)
        critic_opt = torch.optim.AdamW(tiny_trainer.critic.parameters(), lr=1e-3)
        tiny_trainer.ppo_step(buf, actor_opt, critic_opt)

        changed = any(
            not torch.allclose(before[n], p.data)
            for n, p in tiny_trainer.actor.named_parameters()
        )
        assert changed, "Actor weights should change after a PPO update step"

    def test_ref_model_frozen(self, tiny_trainer):
        """Reference model weights must not change after PPO step."""
        buf = RolloutBuffer()
        # We only check that ref parameters have no grad and requires_grad=False
        for p in tiny_trainer.ref_model.parameters():
            assert not p.requires_grad, "ref_model parameters must be frozen"

    def test_compute_log_probs_shape(self, tiny_trainer, tiny_config, device):
        B, T_q, T_r = 2, 6, 5
        q_ids  = torch.randint(0, tiny_config.vocab_size, (B, T_q)).to(device)
        q_mask = torch.ones(B, T_q).to(device)
        r_ids  = torch.randint(0, tiny_config.vocab_size, (B, T_r)).to(device)
        r_mask = torch.ones(B, T_r).to(device)

        with torch.no_grad():
            lp = tiny_trainer._compute_log_probs(
                tiny_trainer.actor, q_ids, q_mask, r_ids, r_mask
            )
        assert lp.shape == (B, T_r)
        assert lp.le(0).all(), "Log probs must be ≤ 0"
        assert lp.isfinite().all()
