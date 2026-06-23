"""
Ablation studies — Phase 3.

Sweeps three dimensions of the PPO hyperparameter space:
  1. KL coefficient β         ∈ {0.01, 0.05, 0.1, 0.2, 0.5}
  2. PPO inner epochs         ∈ {1, 2, 4, 8}
  3. Rollout batch size       ∈ {8, 16, 32, 64}

Each ablation run starts from the SFT checkpoint, trains for
``--ablation_steps`` PPO steps (default 100, much shorter than the full
1 000-step run), and evaluates win rate against the SFT baseline on
``--eval_prompts`` held-out test prompts.

Results are saved to results/tables/ablations.json and a figure to
results/figures/ablations.png.

Usage:
    python scripts/ablations.py \\
        --rm_checkpoint  checkpoints/rm/best_rm.pt \\
        --sft_checkpoint checkpoints/sft/best_sft \\
        [--ablation_steps 100] [--eval_prompts 50] [--sweep kl_coef]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import cycle
from pathlib import Path

import torch
from torch.optim import AdamW
from datasets import load_from_disk
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import compute_win_rate, plot_ablation_results
from src.ppo_trainer import CriticModel, PPOTrainer
from src.reward_model import RewardModel
from src.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Sweep value grids
# ---------------------------------------------------------------------------

KL_COEF_VALUES       = [0.01, 0.05, 0.1, 0.2, 0.5]
N_PPO_EPOCH_VALUES   = [1, 2, 4, 8]
ROLLOUT_BATCH_VALUES = [8, 16, 32, 64]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt_cycle(
    dataset_path: str,
    tokenizer: GPT2Tokenizer,
    batch_size: int,
    max_prompt_len: int = 256,
) -> cycle:
    ds = load_from_disk(dataset_path)
    prompts = ds["train"]["prompt"]
    token_lists = [
        torch.tensor(
            tokenizer.encode(p, add_special_tokens=False, truncation=True, max_length=max_prompt_len),
            dtype=torch.long,
        )
        for p in prompts
    ]
    batches = [token_lists[s : s + batch_size] for s in range(0, len(token_lists), batch_size)]
    return cycle(batches)


def _run_ppo(
    rm_checkpoint: str,
    sft_checkpoint: str,
    data_dir: str,
    device: torch.device,
    n_steps: int,
    kl_coef: float,
    n_ppo_epochs: int,
    rollout_batch_size: int,
) -> tuple[GPT2LMHeadModel, list[dict]]:
    """
    Run a short PPO training loop from the SFT checkpoint.

    Returns (trained_actor, step_history) where step_history is a list of
    per-step metric dicts (rm_score_mean, kl_mean, actor_loss, …).
    """
    reward_model = RewardModel.from_checkpoint(rm_checkpoint)
    actor        = GPT2LMHeadModel.from_pretrained(sft_checkpoint)
    critic       = CriticModel.from_sft_checkpoint(sft_checkpoint)
    ref_model    = GPT2LMHeadModel.from_pretrained(sft_checkpoint)

    tokenizer = GPT2Tokenizer.from_pretrained(sft_checkpoint)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    trainer = PPOTrainer(
        actor=actor, critic=critic,
        ref_model=ref_model, reward_model=reward_model,
        device=device,
        clip_eps=0.2, vf_coef=0.1, entropy_coef=0.01,
        n_ppo_epochs=n_ppo_epochs,
        mini_batch_size=min(4, rollout_batch_size),
        max_grad_norm=1.0,
        kl_coef=kl_coef,
        max_new_tokens=128,
        temperature=1.0,
    )

    actor_opt  = AdamW(actor.parameters(),  lr=1e-5, weight_decay=0.01)
    critic_opt = AdamW(critic.parameters(), lr=1e-5, weight_decay=0.01)

    prompt_iter = _build_prompt_cycle(data_dir, tokenizer, rollout_batch_size)

    history: list[dict] = []
    for step in range(1, n_steps + 1):
        buffer        = RolloutBuffer()
        rollout_stats = trainer.collect_rollouts(next(prompt_iter), buffer)
        buffer.compute_advantages(gamma=1.0, lam=0.95)
        update_stats  = trainer.ppo_step(buffer, actor_opt, critic_opt)
        history.append({"step": step, **rollout_stats, **update_stats})

    return trainer.actor, history


def _eval_win_rate(
    ppo_actor: GPT2LMHeadModel,
    sft_checkpoint: str,
    rm_checkpoint: str,
    data_dir: str,
    device: torch.device,
    n_prompts: int,
) -> dict[str, float]:
    tokenizer = GPT2Tokenizer.from_pretrained(sft_checkpoint)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    sft_model    = GPT2LMHeadModel.from_pretrained(sft_checkpoint)
    reward_model = RewardModel.from_checkpoint(rm_checkpoint)

    ds = load_from_disk(data_dir)
    prompt_ids = [
        torch.tensor(
            tokenizer.encode(p, add_special_tokens=False, truncation=True, max_length=256),
            dtype=torch.long,
        )
        for p in ds["test"]["prompt"][:n_prompts]
    ]

    return compute_win_rate(
        ppo_model=ppo_actor,
        sft_model=sft_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        prompt_ids_list=prompt_ids,
        device=device,
        max_new_tokens=128,
        temperature=1.0,
        batch_size=8,
    )


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def _run_sweep(
    sweep_label: str,
    param_name: str,
    values: list,
    base_kwargs: dict,
    rm_checkpoint: str,
    sft_checkpoint: str,
    data_dir: str,
    device: torch.device,
    n_steps: int,
    n_eval_prompts: int,
) -> list[dict]:
    print(f"\n{'='*60}\nSweep: {sweep_label}\n{'='*60}")
    sweep_results = []

    for v in values:
        kwargs = {**base_kwargs, param_name: v}
        print(f"\n  {param_name}={v}  →  {n_steps} PPO steps…")
        ppo_actor, history = _run_ppo(
            rm_checkpoint=rm_checkpoint,
            sft_checkpoint=sft_checkpoint,
            data_dir=data_dir,
            device=device,
            n_steps=n_steps,
            **kwargs,
        )
        win = _eval_win_rate(
            ppo_actor=ppo_actor,
            sft_checkpoint=sft_checkpoint,
            rm_checkpoint=rm_checkpoint,
            data_dir=data_dir,
            device=device,
            n_prompts=n_eval_prompts,
        )
        final = history[-1] if history else {}
        record = {
            param_name:       v,
            "win_rate":       win["win_rate"],
            "mean_ppo_score": win["mean_ppo_score"],
            "mean_sft_score": win["mean_sft_score"],
            "final_kl":       final.get("kl_mean"),
            "final_rm_score": final.get("rm_score_mean"),
        }
        sweep_results.append(record)
        print(
            f"    win_rate={record['win_rate']:.1%}  "
            f"ppo_score={record['mean_ppo_score']:.4f}  "
            f"kl={record['final_kl']:.4f}"
        )

    return sweep_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PPO ablation studies — Phase 3")
    parser.add_argument("--rm_checkpoint",  required=True)
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--data_dir",  default="data/processed")
    parser.add_argument("--output",    default="results/tables/ablations.json")
    parser.add_argument("--figures_dir", default="results/figures")
    parser.add_argument("--ablation_steps", type=int, default=100,
                        help="PPO steps per run — keep ≤200 for speed")
    parser.add_argument("--eval_prompts",   type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sweep",
        choices=["kl_coef", "n_ppo_epochs", "rollout_batch", "all"],
        default="all",
        help="Which dimension to sweep (default: all)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Ablation steps per run: {args.ablation_steps}  |  "
        f"Eval prompts: {args.eval_prompts}"
    )

    # Base hyperparameters held constant while one dimension is swept.
    BASE = dict(kl_coef=0.1, n_ppo_epochs=4, rollout_batch_size=32)

    shared = dict(
        rm_checkpoint=args.rm_checkpoint,
        sft_checkpoint=args.sft_checkpoint,
        data_dir=args.data_dir,
        device=device,
        n_steps=args.ablation_steps,
        n_eval_prompts=args.eval_prompts,
    )

    results: dict[str, list[dict]] = {}

    if args.sweep in ("kl_coef", "all"):
        results["kl_coef"] = _run_sweep(
            "KL coefficient β", "kl_coef", KL_COEF_VALUES,
            {k: v for k, v in BASE.items() if k != "kl_coef"},
            **shared,
        )

    if args.sweep in ("n_ppo_epochs", "all"):
        results["n_ppo_epochs"] = _run_sweep(
            "PPO inner epochs", "n_ppo_epochs", N_PPO_EPOCH_VALUES,
            {k: v for k, v in BASE.items() if k != "n_ppo_epochs"},
            **shared,
        )

    if args.sweep in ("rollout_batch", "all"):
        results["rollout_batch"] = _run_sweep(
            "Rollout batch size", "rollout_batch_size", ROLLOUT_BATCH_VALUES,
            {k: v for k, v in BASE.items() if k != "rollout_batch_size"},
            **shared,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {args.output}")

    plot_ablation_results(
        results,
        output_path=os.path.join(args.figures_dir, "ablations.png"),
    )


if __name__ == "__main__":
    main()
