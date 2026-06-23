"""
Evaluation suite — Phase 3.

Computes:
  - RM win rate: % of PPO responses that beat SFT baseline according to RM
  - KL divergence from π_ref as a function of training step
  - Reward vs KL trade-off curve (reproduces Gao et al. 2022 Fig 1)
  - Human-readable qualitative examples including failure modes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.reward_model import RewardModel

PAD_TOKEN_ID = 50256

# Shared plot colours (match plot_rm_results.py palette)
BLUE  = "#2E86AB"
RED   = "#E84855"
GRAY  = "#6B6B6B"
GREEN = "#3BB273"
LIGHT = "#F0F4F8"


# ---------------------------------------------------------------------------
# Win-rate
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_win_rate(
    ppo_model: GPT2LMHeadModel,
    sft_model: GPT2LMHeadModel,
    reward_model: RewardModel,
    tokenizer: GPT2Tokenizer,
    prompt_ids_list: list[torch.Tensor],   # list of (T_q,) CPU tensors
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    batch_size: int = 8,
) -> dict[str, float]:
    """
    For each prompt generate one PPO response and one SFT response,
    score both with the RM, count the fraction where PPO wins.

    Left-pads queries (matches the generation convention in PPO training).

    Returns:
        win_rate, mean_ppo_score, mean_sft_score, mean_score_gap
    """
    ppo_model.eval().to(device)
    sft_model.eval().to(device)
    reward_model.eval().to(device)

    ppo_scores_all: list[float] = []
    sft_scores_all: list[float] = []

    for start in range(0, len(prompt_ids_list), batch_size):
        batch = prompt_ids_list[start : start + batch_size]
        B = len(batch)

        max_q = max(t.shape[0] for t in batch)
        query_ids  = torch.full((B, max_q), PAD_TOKEN_ID, dtype=torch.long)
        query_mask = torch.zeros(B, max_q)
        for i, q in enumerate(batch):
            query_ids[i, max_q - q.shape[0]:]  = q
            query_mask[i, max_q - q.shape[0]:] = 1.0
        query_ids  = query_ids.to(device)
        query_mask = query_mask.to(device)

        def _generate_and_score(model: GPT2LMHeadModel) -> torch.Tensor:
            out = model.generate(
                input_ids=query_ids,
                attention_mask=query_mask,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=PAD_TOKEN_ID,
                eos_token_id=PAD_TOKEN_ID,
            )
            resp_ids  = out[:, max_q:]
            eos_cumsum  = (resp_ids == PAD_TOKEN_ID).long().cumsum(dim=1)
            resp_mask   = (eos_cumsum <= 1).float()
            full_ids    = torch.cat([query_ids,  resp_ids],  dim=1)
            full_mask   = torch.cat([query_mask, resp_mask], dim=1)
            return reward_model(full_ids, full_mask)  # (B,)

        ppo_scores_all.extend(_generate_and_score(ppo_model).cpu().tolist())
        sft_scores_all.extend(_generate_and_score(sft_model).cpu().tolist())

    ppo_t = torch.tensor(ppo_scores_all)
    sft_t = torch.tensor(sft_scores_all)

    return {
        "win_rate":       (ppo_t > sft_t).float().mean().item(),
        "mean_ppo_score": ppo_t.mean().item(),
        "mean_sft_score": sft_t.mean().item(),
        "mean_score_gap": (ppo_t - sft_t).mean().item(),
    }


# ---------------------------------------------------------------------------
# KL-reward curve
# ---------------------------------------------------------------------------

def extract_kl_reward_curve(
    history: list[dict[str, Any]],
) -> tuple[list[int], list[float], list[float]]:
    """
    Pull (steps, rm_scores, kl_values) out of a PPO training history list.

    Each entry must have keys: step, rm_score_mean, kl_mean.
    """
    steps   = [h["step"]          for h in history]
    rewards = [h["rm_score_mean"] for h in history]
    kls     = [h["kl_mean"]       for h in history]
    return steps, rewards, kls


# ---------------------------------------------------------------------------
# Qualitative examples
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_qualitative_examples(
    ppo_model: GPT2LMHeadModel,
    sft_model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompts: list[str],
    device: torch.device,
    reward_model: RewardModel | None = None,
    n_examples: int = 10,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> list[dict]:
    """
    Generate side-by-side PPO vs SFT responses for qualitative analysis.

    Optionally scores each response with the RM if reward_model is provided.

    Returns list of dicts with keys:
        prompt, ppo_response, sft_response,
        ppo_score (optional), sft_score (optional)
    """
    ppo_model.eval().to(device)
    sft_model.eval().to(device)
    if reward_model is not None:
        reward_model.eval().to(device)

    tokenizer.padding_side = "left"
    tokenizer.pad_token    = tokenizer.eos_token

    examples: list[dict] = []

    for prompt in prompts[:n_examples]:
        enc  = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        T_q  = ids.shape[1]

        def _gen(model: GPT2LMHeadModel) -> str:
            out = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=PAD_TOKEN_ID,
                eos_token_id=PAD_TOKEN_ID,
            )
            resp = out[0, T_q:]
            resp = resp[resp != PAD_TOKEN_ID]
            return tokenizer.decode(resp, skip_special_tokens=True)

        ppo_text = _gen(ppo_model)
        sft_text = _gen(sft_model)

        entry: dict = {"prompt": prompt, "ppo_response": ppo_text, "sft_response": sft_text}

        if reward_model is not None:
            for score_key, text in [("ppo_score", ppo_text), ("sft_score", sft_text)]:
                full_enc = tokenizer(
                    prompt + text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)
                entry[score_key] = reward_model(
                    full_enc["input_ids"], full_enc["attention_mask"]
                ).item()

        examples.append(entry)

    return examples


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reward_kl_frontier(
    steps: list[int],
    rewards: list[float],
    kls: list[float],
    output_path: str,
) -> None:
    """
    Scatter plot of RM reward vs KL divergence coloured by training step.

    Reproduces the Gao et al. 2022 Fig 1 style: KL on x-axis, RM score on
    y-axis. Points darken over training — the frontier traces the reward-KL
    Pareto curve. A policy that climbs reward without increasing KL is ideal;
    one that drives KL up without reward gain is reward-hacking.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(LIGHT)
    ax.grid(True, color="white", linewidth=1.2, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    sc = ax.scatter(
        kls, rewards,
        c=steps, cmap="viridis", s=20, alpha=0.75, edgecolors="none", zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Training step", fontsize=10)

    ax.set_xlabel("KL divergence from π_ref (nats)", fontsize=12)
    ax.set_ylabel("RM score", fontsize=12)
    ax.set_title(
        "Reward vs KL Trade-off  ·  Gao et al. 2022 Fig. 1 style",
        fontsize=13, fontweight="bold", pad=10,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"Saved reward-KL frontier to {output_path}")


def plot_ablation_results(
    ablation_results: dict[str, list[dict]],
    output_path: str,
) -> None:
    """
    One subplot per ablation sweep showing win rate as a function of the swept parameter.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping ablation plot.")
        return

    sweeps = list(ablation_results.keys())
    n = len(sweeps)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    label_map = {
        "kl_coef":        ("KL coefficient β",      "kl_coef"),
        "n_ppo_epochs":   ("PPO inner epochs",       "n_ppo_epochs"),
        "rollout_batch":  ("Rollout batch size",     "rollout_batch_size"),
    }

    for ax, sweep_name in zip(axes, sweeps):
        records = ablation_results[sweep_name]
        x_label, x_key = label_map.get(sweep_name, (sweep_name, sweep_name))
        xs = [r[x_key] for r in records]
        ys = [r["win_rate"] * 100 for r in records]

        ax.plot(xs, ys, "o-", color=BLUE, linewidth=2.5, markersize=7, zorder=3)
        ax.axhline(50, color=GRAY, linestyle=":", linewidth=1.5, label="50% baseline", zorder=2)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Win rate vs SFT (%)", fontsize=11)
        ax.set_title(f"Ablation: {x_label}", fontsize=12, fontweight="bold", pad=8)
        ax.set_facecolor(LIGHT)
        ax.grid(True, color="white", linewidth=1.2, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.suptitle(
        "Ablation Studies: Win Rate vs PPO Hyperparameters",
        fontsize=12, y=1.02, color=GRAY,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"Saved ablation plot to {output_path}")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_win_rate_table(results: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved win-rate table to {output_path}")


def save_qualitative_examples(examples: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(examples)} qualitative examples to {output_path}")
