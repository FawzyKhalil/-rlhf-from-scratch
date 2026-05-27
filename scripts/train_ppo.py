"""
PPO-RLHF training loop — Phase 2.

Requires Phase 1 checkpoints:
    --rm_checkpoint   checkpoints/rm/best_rm.pt
    --sft_checkpoint  checkpoints/sft/best_sft

Reads hyper-parameters from configs/ppo_config.yaml (override with --config).

Usage:
    python scripts/train_ppo.py \\
        --rm_checkpoint  checkpoints/rm/best_rm.pt \\
        --sft_checkpoint checkpoints/sft/best_sft \\
        [--config configs/ppo_config.yaml]

Metrics logged each step (stdout + JSON history):
    rm_score_mean  — mean RM score across the rollout batch
    kl_mean        — mean per-sequence KL (monitor: should stay below ~5 nats)
    actor_loss     — clipped surrogate loss
    critic_loss    — value function MSE loss
    entropy        — mean token entropy (higher = more exploration)
    approx_kl      — approx KL between old and new policy (PPO diagnostic)
    response_len   — mean response length in tokens
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Allow running as `python scripts/train_ppo.py` from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ppo_trainer import CriticModel, PPOTrainer
from src.reward_model import RewardModel
from src.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_prompt_loader(
    dataset_path: str,
    tokenizer: GPT2Tokenizer,
    rollout_batch_size: int,
    max_prompt_len: int = 256,
) -> cycle:
    """
    Return a cycling iterator that yields lists of tokenized prompts.

    We use only the ``prompt`` field (no responses) because PPO generates its
    own responses. The iterator cycles forever so the training loop never
    needs to worry about epoch boundaries.
    """
    ds = load_from_disk(dataset_path)
    train_ds = ds["train"]

    prompts = train_ds["prompt"]  # list[str]
    tokenizer.padding_side = "left"  # required for batched generation

    # Tokenise all prompts once; store as CPU tensors.
    token_lists: list[torch.Tensor] = []
    for p in prompts:
        ids = tokenizer.encode(
            p,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_len,
        )
        token_lists.append(torch.tensor(ids, dtype=torch.long))

    # Batch them into fixed-size lists.
    batches: list[list[torch.Tensor]] = []
    for start in range(0, len(token_lists), rollout_batch_size):
        batches.append(token_lists[start : start + rollout_batch_size])

    return cycle(batches)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PPO-RLHF training — Phase 2")
    parser.add_argument("--rm_checkpoint",  required=True, help="Path to best_rm.pt")
    parser.add_argument("--sft_checkpoint", required=True, help="Path to best_sft dir")
    parser.add_argument(
        "--config",
        default="configs/ppo_config.yaml",
        help="YAML config (default: configs/ppo_config.yaml)",
    )
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Config ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Tokenizer ---
    tokenizer = GPT2Tokenizer.from_pretrained(args.sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Load models ---
    print("Loading reward model (frozen)…")
    reward_model = RewardModel.from_checkpoint(args.rm_checkpoint)

    print("Loading actor from SFT checkpoint…")
    actor = GPT2LMHeadModel.from_pretrained(args.sft_checkpoint)

    print("Loading critic (transformer from SFT, new value head)…")
    critic = CriticModel.from_sft_checkpoint(args.sft_checkpoint)

    print("Loading reference model (frozen SFT)…")
    ref_model = GPT2LMHeadModel.from_pretrained(args.sft_checkpoint)

    # --- Optimisers ---
    actor_optimizer = AdamW(
        actor.parameters(),
        lr=cfg["training"]["lr_actor"],
        weight_decay=0.01,
    )
    critic_optimizer = AdamW(
        critic.parameters(),
        lr=cfg["training"]["lr_critic"],
        weight_decay=0.01,
    )

    total_steps   = cfg["training"]["total_steps"]
    warmup_steps  = cfg["training"]["warmup_steps"]

    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    critic_scheduler = LinearLR(
        critic_optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # --- PPO trainer ---
    trainer = PPOTrainer(
        actor=actor,
        critic=critic,
        ref_model=ref_model,
        reward_model=reward_model,
        device=device,
        clip_eps=cfg["ppo"]["clip_range"],
        vf_coef=0.1,
        entropy_coef=0.01,
        n_ppo_epochs=cfg["ppo"]["epochs"],
        mini_batch_size=cfg["ppo"]["mini_batch_size"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        kl_coef=cfg["reward"]["kl_coef"],
        max_new_tokens=cfg["rollout"]["max_new_tokens"],
        temperature=cfg["rollout"]["temperature"],
    )

    # --- Data ---
    prompt_iter = build_prompt_loader(
        args.data_dir,
        tokenizer,
        rollout_batch_size=cfg["rollout"]["batch_size"],
    )

    # --- Checkpoint dir ---
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    log_interval  = cfg["training"]["log_interval"]
    save_interval = cfg["training"]["save_interval"]
    history: list[dict] = []

    best_rm_score = float("-inf")

    print(f"\nStarting PPO training for {total_steps} steps…\n")
    print(
        f"{'Step':>6}  {'RM score':>10}  {'KL':>8}  "
        f"{'Actor L':>10}  {'Critic L':>10}  {'Entropy':>10}  {'Resp len':>9}"
    )
    print("-" * 78)

    for step in range(1, total_steps + 1):
        # ---- Rollout collection ----
        prompt_batch = next(prompt_iter)
        buffer = RolloutBuffer()

        rollout_stats = trainer.collect_rollouts(prompt_batch, buffer)

        # ---- GAE advantages ----
        buffer.compute_advantages(gamma=1.0, lam=0.95)

        # ---- PPO update ----
        update_stats = trainer.ppo_step(
            buffer, actor_optimizer, critic_optimizer
        )

        # Warmup schedulers advance per step (not per epoch)
        if step <= warmup_steps:
            actor_scheduler.step()
            critic_scheduler.step()

        # ---- Logging ----
        stats = {**rollout_stats, **update_stats, "step": step}
        history.append(stats)

        if step % log_interval == 0:
            print(
                f"{step:>6}  "
                f"{stats['rm_score_mean']:>10.4f}  "
                f"{stats['kl_mean']:>8.4f}  "
                f"{stats['actor_loss']:>10.4f}  "
                f"{stats['critic_loss']:>10.4f}  "
                f"{stats['entropy']:>10.4f}  "
                f"{stats['response_len']:>9.1f}"
            )

        # ---- Checkpointing ----
        if stats["rm_score_mean"] > best_rm_score:
            best_rm_score = stats["rm_score_mean"]
            trainer.save_checkpoint(ckpt_dir, step=0)  # step=0 → "best"

        if step % save_interval == 0:
            trainer.save_checkpoint(ckpt_dir, step=step)
            with open(os.path.join(ckpt_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=2)

    # --- Final save ---
    trainer.save_checkpoint(ckpt_dir, step=total_steps)
    with open(os.path.join(ckpt_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save best actor in HuggingFace format for downstream evaluation
    best_actor_path = os.path.join(ckpt_dir, "best_ppo_actor")
    actor.save_pretrained(best_actor_path)
    tokenizer.save_pretrained(best_actor_path)
    print(f"\nBest actor saved to {best_actor_path}")
    print(f"Best RM score achieved: {best_rm_score:.4f}")


if __name__ == "__main__":
    main()
