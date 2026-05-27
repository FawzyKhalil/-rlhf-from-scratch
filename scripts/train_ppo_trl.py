"""
PPO-RLHF with TRL's PPOTrainer — validation baseline (Phase 2).

This script reproduces the same experiment as ``train_ppo.py`` using TRL's
reference implementation. The purpose is to validate our from-scratch version:
the two training curves (RM score, KL) should converge to the same region
given identical hyper-parameters and the same Phase-1 checkpoints.

Usage:
    pip install trl>=0.7.0
    python scripts/train_ppo_trl.py \\
        --rm_checkpoint  checkpoints/rm/best_rm.pt \\
        --sft_checkpoint checkpoints/sft/best_sft \\
        [--config configs/ppo_config.yaml]

Key differences from TRL defaults that we align to match train_ppo.py:
    - Same kl_coef, clip_range, batch_size, mini_batch_size
    - GPT-2 backbone loaded from the SFT checkpoint for both actor and ref
    - Reward function calls the Phase-1 RewardModel (not a separate pipeline)

TRL wraps the actor as AutoModelForCausalLMWithValueHead, which attaches a
per-token value head on top of the LM backbone — the same architecture we
implement manually in CriticModel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import cycle

import torch
import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer, GPT2Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reward_model import RewardModel

try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
except ImportError as e:
    raise ImportError(
        "TRL is required for this validation script.\n"
        "  pip install trl>=0.7.0"
    ) from e


# ---------------------------------------------------------------------------
# Reward function (wraps our Phase-1 RM)
# ---------------------------------------------------------------------------

def make_reward_fn(reward_model: RewardModel, tokenizer, device: torch.device):
    """
    Return a callable that takes (query_tensors, response_tensors) → list[Tensor].

    TRL's PPOTrainer expects a reward function with this signature where each
    element of the returned list is a scalar Tensor for the corresponding pair.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    @torch.no_grad()
    def reward_fn(
        query_tensors: list[torch.Tensor],
        response_tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        rewards = []
        for q, r in zip(query_tensors, response_tensors):
            full_ids  = torch.cat([q, r], dim=0).unsqueeze(0).to(device)   # (1, T)
            attn_mask = (full_ids != pad_id).float()
            score = reward_model(full_ids, attn_mask).squeeze()            # scalar
            rewards.append(score)
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def load_prompts(
    data_dir: str,
    tokenizer,
    rollout_batch_size: int,
    max_prompt_len: int = 256,
) -> cycle:
    ds = load_from_disk(data_dir)
    prompts = ds["train"]["prompt"]
    tokenizer.padding_side = "left"

    token_lists: list[torch.Tensor] = []
    for p in prompts:
        ids = tokenizer.encode(
            p,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_len,
        )
        token_lists.append(torch.tensor(ids, dtype=torch.long))

    batches = [
        token_lists[s : s + rollout_batch_size]
        for s in range(0, len(token_lists), rollout_batch_size)
    ]
    return cycle(batches)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PPO-RLHF via TRL — validation")
    parser.add_argument("--rm_checkpoint",  required=True)
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--config", default="configs/ppo_config.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Tokenizer ---
    tokenizer = GPT2Tokenizer.from_pretrained(args.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- TRL PPOConfig (mirrors our from-scratch hyper-parameters) ---
    ppo_config = PPOConfig(
        model_name=args.sft_checkpoint,
        learning_rate=cfg["training"]["lr_actor"],
        batch_size=cfg["rollout"]["batch_size"],
        mini_batch_size=cfg["ppo"]["mini_batch_size"],
        ppo_epochs=cfg["ppo"]["epochs"],
        cliprange=cfg["ppo"]["clip_range"],
        cliprange_value=cfg["ppo"]["value_clip_range"],
        kl_coef=cfg["reward"]["kl_coef"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        seed=args.seed,
        log_with=None,  # set to "wandb" to enable W&B logging
    )

    # --- Models ---
    # TRL wraps the actor in AutoModelForCausalLMWithValueHead which adds a
    # per-token scalar value head — exactly the architecture we built manually.
    print("Loading actor + value head from SFT checkpoint…")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft_checkpoint)
    model = model.to(device)

    print("Loading reference model (frozen SFT)…")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft_checkpoint)
    ref_model = ref_model.to(device)

    print("Loading reward model (frozen Phase-1 checkpoint)…")
    reward_model = RewardModel.from_checkpoint(args.rm_checkpoint).to(device)
    reward_model.eval()
    reward_fn = make_reward_fn(reward_model, tokenizer, device)

    # --- TRL trainer ---
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    # --- Data ---
    prompt_iter = load_prompts(
        args.data_dir,
        tokenizer,
        rollout_batch_size=cfg["rollout"]["batch_size"],
    )

    total_steps = cfg["training"]["total_steps"]
    log_interval = cfg["training"]["log_interval"]
    ckpt_dir = os.path.join(cfg["training"]["checkpoint_dir"], "trl")
    os.makedirs(ckpt_dir, exist_ok=True)

    history: list[dict] = []
    print(f"\nStarting TRL PPO training for {total_steps} steps…\n")

    for step in range(1, total_steps + 1):
        query_tensors = next(prompt_iter)  # list of (T_q_i,) CPU tensors

        # Generate responses
        response_tensors = trainer.generate(
            query_tensors,
            max_new_tokens=cfg["rollout"]["max_new_tokens"],
            do_sample=True,
            temperature=cfg["rollout"]["temperature"],
            pad_token_id=tokenizer.eos_token_id,
        )

        # Score responses
        rewards = reward_fn(query_tensors, response_tensors)

        # PPO update
        stats = trainer.step(query_tensors, response_tensors, rewards)

        step_stats = {
            "step":        step,
            "rm_score":    torch.stack(rewards).mean().item(),
            "kl":          stats.get("objective/kl", float("nan")),
            "actor_loss":  stats.get("ppo/loss/policy", float("nan")),
            "critic_loss": stats.get("ppo/loss/value", float("nan")),
            "entropy":     stats.get("objective/entropy", float("nan")),
        }
        history.append(step_stats)

        if step % log_interval == 0:
            print(
                f"step {step:4d} | "
                f"rm_score={step_stats['rm_score']:.4f} | "
                f"kl={step_stats['kl']:.4f} | "
                f"actor_loss={step_stats['actor_loss']:.4f}"
            )

    # --- Save ---
    model.save_pretrained(os.path.join(ckpt_dir, "final_actor"))
    tokenizer.save_pretrained(os.path.join(ckpt_dir, "final_actor"))
    with open(os.path.join(ckpt_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTRL training complete. Saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
