"""
Unit tests for Phase 3: synthetic preferences, evaluation suite.

All tests use tiny GPT-2 configs from conftest.py — no weight downloads.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from src.evaluate import (
    compute_win_rate,
    extract_kl_reward_curve,
    generate_qualitative_examples,
    plot_reward_kl_frontier,
    save_qualitative_examples,
    save_win_rate_table,
)
from src.reward_model import RewardModel
from data.synthetic_preferences import generate_synthetic_preferences


# ---------------------------------------------------------------------------
# Shared tiny-model helpers
# ---------------------------------------------------------------------------

def make_lm(cfg: GPT2Config) -> GPT2LMHeadModel:
    return GPT2LMHeadModel(cfg)


def make_rm(cfg: GPT2Config) -> RewardModel:
    return RewardModel(model_name=cfg)


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_config):
    """
    GPT-2 tokenizer whose IDs are clamped to tiny_config.vocab_size via modulo.

    The real tokenizer produces IDs up to 50256; tiny models only have
    vocab_size=1000 embeddings, so raw GPT-2 IDs would cause IndexError.
    Clamping preserves the encode/decode interface while keeping IDs in range.
    """
    base = GPT2Tokenizer.from_pretrained("gpt2")
    base.pad_token = base.eos_token
    vocab_size = tiny_config.vocab_size

    class _ClampedTokenizer:
        def __init__(self):
            self.pad_token     = base.pad_token
            self.eos_token     = base.eos_token
            self._padding_side = "left"

        @property
        def padding_side(self):
            return self._padding_side

        @padding_side.setter
        def padding_side(self, v):
            self._padding_side = v
            base.padding_side  = v

        def encode(self, text, **kwargs):
            return [i % vocab_size for i in base.encode(text, **kwargs)]

        def __call__(self, text, return_tensors=None, **kwargs):
            base.padding_side = self._padding_side
            out = base(text, return_tensors=return_tensors, **kwargs)
            if return_tensors == "pt":
                out["input_ids"] = out["input_ids"] % vocab_size
            elif isinstance(out.get("input_ids"), list):
                ids = out["input_ids"]
                if ids and isinstance(ids[0], list):
                    out["input_ids"] = [[i % vocab_size for i in row] for row in ids]
                else:
                    out["input_ids"] = [i % vocab_size for i in ids]
            return out

        def decode(self, ids, **kwargs):
            return base.decode(ids, **kwargs)

    return _ClampedTokenizer()


# ---------------------------------------------------------------------------
# Tests: synthetic preferences
# ---------------------------------------------------------------------------

class TestSyntheticPreferences:

    def test_output_length(self, tiny_config, device, tiny_tokenizer):
        """One record per prompt."""
        actor = make_lm(tiny_config)
        rm    = make_rm(tiny_config)
        prompts = ["Hello world", "How are you"]
        records = generate_synthetic_preferences(
            actor, rm, tiny_tokenizer, prompts, device, K=2, max_new_tokens=4, batch_size=2
        )
        assert len(records) == len(prompts)

    def test_required_keys(self, tiny_config, device, tiny_tokenizer):
        actor = make_lm(tiny_config)
        rm    = make_rm(tiny_config)
        records = generate_synthetic_preferences(
            actor, rm, tiny_tokenizer, ["Hi"], device, K=2, max_new_tokens=4, batch_size=1
        )
        for key in ("prompt", "chosen_response", "rejected_response",
                    "chosen_score", "rejected_score"):
            assert key in records[0], f"Missing key: {key}"

    def test_chosen_score_geq_rejected(self, tiny_config, device, tiny_tokenizer):
        """Chosen must always have a score ≥ rejected (by construction)."""
        actor = make_lm(tiny_config)
        rm    = make_rm(tiny_config)
        records = generate_synthetic_preferences(
            actor, rm, tiny_tokenizer, ["Hello", "World", "Test"], device,
            K=4, max_new_tokens=4, batch_size=2,
        )
        for r in records:
            assert r["chosen_score"] >= r["rejected_score"], (
                f"chosen_score ({r['chosen_score']:.4f}) < rejected_score "
                f"({r['rejected_score']:.4f})"
            )

    def test_scores_are_floats(self, tiny_config, device, tiny_tokenizer):
        actor = make_lm(tiny_config)
        rm    = make_rm(tiny_config)
        records = generate_synthetic_preferences(
            actor, rm, tiny_tokenizer, ["Hi"], device, K=2, max_new_tokens=4, batch_size=1
        )
        assert isinstance(records[0]["chosen_score"], float)
        assert isinstance(records[0]["rejected_score"], float)


# ---------------------------------------------------------------------------
# Tests: extract_kl_reward_curve
# ---------------------------------------------------------------------------

class TestExtractKLRewardCurve:

    def test_length_matches_history(self):
        history = [
            {"step": 1, "rm_score_mean": 0.1, "kl_mean": 0.5, "actor_loss": 0.3},
            {"step": 2, "rm_score_mean": 0.2, "kl_mean": 0.6, "actor_loss": 0.2},
            {"step": 3, "rm_score_mean": 0.3, "kl_mean": 0.7, "actor_loss": 0.1},
        ]
        steps, rewards, kls = extract_kl_reward_curve(history)
        assert len(steps) == len(rewards) == len(kls) == 3

    def test_values_correct(self):
        history = [
            {"step": 10, "rm_score_mean": 1.5, "kl_mean": 2.0},
            {"step": 20, "rm_score_mean": 2.5, "kl_mean": 3.0},
        ]
        steps, rewards, kls = extract_kl_reward_curve(history)
        assert steps   == [10, 20]
        assert rewards == [1.5, 2.5]
        assert kls     == [2.0, 3.0]

    def test_empty_history(self):
        steps, rewards, kls = extract_kl_reward_curve([])
        assert steps == rewards == kls == []


# ---------------------------------------------------------------------------
# Tests: compute_win_rate
# ---------------------------------------------------------------------------

class TestComputeWinRate:
    # compute_win_rate takes pre-tokenized (T_q,) tensors — no tokenizer call inside.
    # We still pass tiny_tokenizer to match the function signature.

    def test_return_keys(self, tiny_config, device, tiny_tokenizer):
        ppo = make_lm(tiny_config)
        sft = make_lm(tiny_config)
        rm  = make_rm(tiny_config)
        prompts = [torch.randint(0, tiny_config.vocab_size, (5,)) for _ in range(4)]
        result = compute_win_rate(
            ppo, sft, rm, tiny_tokenizer, prompts, device,
            max_new_tokens=4, temperature=1.0, batch_size=2,
        )
        for k in ("win_rate", "mean_ppo_score", "mean_sft_score", "mean_score_gap"):
            assert k in result, f"Missing key: {k}"

    def test_win_rate_in_range(self, tiny_config, device, tiny_tokenizer):
        ppo = make_lm(tiny_config)
        sft = make_lm(tiny_config)
        rm  = make_rm(tiny_config)
        prompts = [torch.randint(0, tiny_config.vocab_size, (5,)) for _ in range(4)]
        result = compute_win_rate(
            ppo, sft, rm, tiny_tokenizer, prompts, device,
            max_new_tokens=4, temperature=1.0, batch_size=4,
        )
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_score_gap_equals_ppo_minus_sft(self, tiny_config, device, tiny_tokenizer):
        ppo = make_lm(tiny_config)
        sft = make_lm(tiny_config)
        rm  = make_rm(tiny_config)
        prompts = [torch.randint(0, tiny_config.vocab_size, (5,)) for _ in range(2)]
        result = compute_win_rate(
            ppo, sft, rm, tiny_tokenizer, prompts, device,
            max_new_tokens=4, temperature=1.0, batch_size=2,
        )
        expected_gap = result["mean_ppo_score"] - result["mean_sft_score"]
        assert abs(result["mean_score_gap"] - expected_gap) < 1e-5

    def test_identical_models_win_rate_near_half(self, tiny_config, device, tiny_tokenizer):
        """When both models are identical, greedy decoding produces equal scores → gap ≈ 0."""
        model = make_lm(tiny_config)
        rm    = make_rm(tiny_config)
        prompts = [torch.randint(0, tiny_config.vocab_size, (5,)) for _ in range(6)]
        result = compute_win_rate(
            model, model, rm, tiny_tokenizer, prompts, device,
            max_new_tokens=4, temperature=0.0,
            batch_size=6,
        )
        # Identical deterministic outputs → same RM score → score gap == 0
        assert result["mean_score_gap"] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Tests: generate_qualitative_examples
# ---------------------------------------------------------------------------

class TestGenerateQualitativeExamples:

    def test_output_count(self, tiny_config, device, tiny_tokenizer):
        ppo = make_lm(tiny_config)
        sft = make_lm(tiny_config)
        prompts = ["Hello", "World", "Test prompt"]
        examples = generate_qualitative_examples(
            ppo, sft, tiny_tokenizer, prompts, device,
            n_examples=2, max_new_tokens=4,
        )
        assert len(examples) == 2

    def test_required_keys_no_rm(self, tiny_config, device, tiny_tokenizer):
        ppo = make_lm(tiny_config)
        sft = make_lm(tiny_config)
        examples = generate_qualitative_examples(
            ppo, sft, tiny_tokenizer, ["Hi there"], device,
            n_examples=1, max_new_tokens=4,
        )
        for k in ("prompt", "ppo_response", "sft_response"):
            assert k in examples[0]
        assert "ppo_score" not in examples[0]

    def test_scores_present_with_rm(self, tiny_config, device, tiny_tokenizer):
        ppo = make_lm(tiny_config)
        sft = make_lm(tiny_config)
        rm  = make_rm(tiny_config)
        examples = generate_qualitative_examples(
            ppo, sft, tiny_tokenizer, ["Hello world"], device,
            reward_model=rm, n_examples=1, max_new_tokens=4,
        )
        assert "ppo_score" in examples[0]
        assert "sft_score" in examples[0]
        assert isinstance(examples[0]["ppo_score"], float)

    def test_prompt_preserved(self, tiny_config, device, tiny_tokenizer):
        ppo = make_lm(tiny_config)
        sft = make_lm(tiny_config)
        prompt = "A unique test prompt"
        examples = generate_qualitative_examples(
            ppo, sft, tiny_tokenizer, [prompt], device,
            n_examples=1, max_new_tokens=4,
        )
        assert examples[0]["prompt"] == prompt


# ---------------------------------------------------------------------------
# Tests: I/O helpers
# ---------------------------------------------------------------------------

class TestIOHelpers:

    def test_save_win_rate_table(self, tmp_path):
        data = {"win_rate": 0.65, "mean_ppo_score": 1.2}
        out  = str(tmp_path / "sub" / "win_rate.json")
        save_win_rate_table(data, out)
        with open(out) as f:
            loaded = json.load(f)
        assert loaded["win_rate"] == pytest.approx(0.65)

    def test_save_qualitative_examples(self, tmp_path):
        examples = [{"prompt": "Hi", "ppo_response": "A", "sft_response": "B"}]
        out = str(tmp_path / "examples.json")
        save_qualitative_examples(examples, out)
        with open(out, encoding="utf-8") as f:
            loaded = json.load(f)
        assert len(loaded) == 1
        assert loaded[0]["prompt"] == "Hi"

    def test_plot_reward_kl_frontier_no_crash(self, tmp_path):
        """Plotting must not raise even with minimal data."""
        steps   = [1, 2, 3]
        rewards = [0.1, 0.2, 0.3]
        kls     = [0.5, 0.6, 0.7]
        out = str(tmp_path / "frontier.png")
        plot_reward_kl_frontier(steps, rewards, kls, out)
        assert os.path.exists(out)
