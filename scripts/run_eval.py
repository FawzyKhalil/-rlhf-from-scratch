"""
Evaluation script — Phase 3.

Runs the full evaluation suite against the PPO and SFT models:
  1. RM win rate (PPO vs SFT on held-out test prompts)
  2. Reward vs KL trade-off plot from PPO training history
  3. Qualitative side-by-side response examples

Usage:
    python scripts/run_eval.py [--config configs/eval_config.yaml]

Outputs (paths set in eval_config.yaml):
    results/figures/reward_vs_kl.png
    results/tables/win_rate.json
    results/tables/qualitative_examples.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml
from datasets import load_from_disk
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import (
    compute_win_rate,
    extract_kl_reward_curve,
    generate_qualitative_examples,
    plot_reward_kl_frontier,
    save_qualitative_examples,
    save_win_rate_table,
)
from src.reward_model import RewardModel


def main() -> None:
    parser = argparse.ArgumentParser(description="RLHF evaluation — Phase 3")
    parser.add_argument("--config",  default="configs/eval_config.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument(
        "--history",
        default="checkpoints/ppo/training_history.json",
        help="PPO training history JSON written by train_ppo.py",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load models ---
    print("Loading reward model…")
    reward_model = RewardModel.from_checkpoint(cfg["models"]["reward_model"])

    print("Loading PPO actor…")
    ppo_actor = GPT2LMHeadModel.from_pretrained(cfg["models"]["ppo"])

    print("Loading SFT baseline…")
    sft_model = GPT2LMHeadModel.from_pretrained(cfg["models"]["sft"])

    tokenizer = GPT2Tokenizer.from_pretrained(cfg["models"]["sft"])
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- Load test prompts ---
    print("Loading test split…")
    ds         = load_from_disk(args.data_dir)
    test_text  = ds["test"]["prompt"][: cfg["evaluation"]["n_prompts"]]

    prompt_ids_list = [
        torch.tensor(
            tokenizer.encode(p, add_special_tokens=False, truncation=True, max_length=256),
            dtype=torch.long,
        )
        for p in test_text
    ]

    # --- 1. Win rate ---
    print(f"\nComputing win rate on {len(prompt_ids_list)} prompts…")
    win_results = compute_win_rate(
        ppo_model=ppo_actor,
        sft_model=sft_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        prompt_ids_list=prompt_ids_list,
        device=device,
        max_new_tokens=cfg["evaluation"]["max_new_tokens"],
        temperature=cfg["evaluation"]["temperature"],
    )
    print(
        f"  Win rate:      {win_results['win_rate']:.1%}\n"
        f"  PPO RM score:  {win_results['mean_ppo_score']:.4f}\n"
        f"  SFT RM score:  {win_results['mean_sft_score']:.4f}\n"
        f"  Score gap:     {win_results['mean_score_gap']:.4f}"
    )
    save_win_rate_table(
        win_results,
        os.path.join(cfg["outputs"]["tables_dir"], "win_rate.json"),
    )

    # --- 2. Reward vs KL plot ---
    if os.path.exists(args.history):
        print("\nPlotting reward vs KL frontier…")
        with open(args.history) as f:
            history = json.load(f)
        steps, rewards, kls = extract_kl_reward_curve(history)
        plot_reward_kl_frontier(
            steps, rewards, kls,
            output_path=os.path.join(cfg["outputs"]["figures_dir"], "reward_vs_kl.png"),
        )
    else:
        print(f"\nNo training history at {args.history} — skipping KL plot.")

    # --- 3. Qualitative examples ---
    print("\nGenerating qualitative examples…")
    examples = generate_qualitative_examples(
        ppo_model=ppo_actor,
        sft_model=sft_model,
        tokenizer=tokenizer,
        prompts=test_text,
        device=device,
        reward_model=reward_model,
        n_examples=10,
        max_new_tokens=cfg["evaluation"]["max_new_tokens"],
        temperature=cfg["evaluation"]["temperature"],
    )
    save_qualitative_examples(
        examples,
        os.path.join(cfg["outputs"]["tables_dir"], "qualitative_examples.json"),
    )

    if examples:
        ex = examples[0]
        ppo_score_str = f"{ex['ppo_score']:.4f}" if "ppo_score" in ex else "N/A"
        sft_score_str = f"{ex['sft_score']:.4f}" if "sft_score" in ex else "N/A"
        print("\n" + "=" * 70)
        print("SAMPLE EXAMPLE")
        print("=" * 70)
        print(f"PROMPT:\n{ex['prompt'][:200]}")
        print(f"\nPPO (score={ppo_score_str}):\n{ex['ppo_response'][:300]}")
        print(f"\nSFT (score={sft_score_str}):\n{ex['sft_response'][:300]}")
        print("=" * 70)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
