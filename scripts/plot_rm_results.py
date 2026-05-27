"""
Generate Fig 1 plots for the reward model.

  rm_val_accuracy.png       — val accuracy + train accuracy over epochs,
                               with 50% random baseline and 65% target line
  rm_reward_distributions.png — chosen vs rejected score distributions
                               at each training epoch (shows separation growing)

Called automatically at the end of train_rm.py, or run standalone:
    python scripts/plot_rm_results.py [--checkpoint_dir ...] [--data_dir ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.reward_model import RewardModel

GPT2_PAD_ID = 50256

# ---------------------------------------------------------------------------
# Shared dataset utilities (mirrors train_rm.py to avoid circular import)
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        return {
            "chosen_input_ids":      torch.tensor(row["chosen_input_ids"],      dtype=torch.long),
            "chosen_attention_mask": torch.tensor(row["chosen_attention_mask"], dtype=torch.long),
            "rejected_input_ids":    torch.tensor(row["rejected_input_ids"],    dtype=torch.long),
            "rejected_attention_mask": torch.tensor(row["rejected_attention_mask"], dtype=torch.long),
        }


def _collate(batch: list[dict]) -> dict:
    def pad(seqs: list[torch.Tensor]):
        max_len = max(s.shape[0] for s in seqs)
        ids  = torch.full((len(seqs), max_len), GPT2_PAD_ID, dtype=torch.long)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :s.shape[0]] = s
            mask[i, :s.shape[0]] = 1
        return ids, mask

    c_ids, c_mask = pad([b["chosen_input_ids"]   for b in batch])
    r_ids, r_mask = pad([b["rejected_input_ids"] for b in batch])
    return {
        "chosen_input_ids":      c_ids,
        "chosen_attention_mask": c_mask,
        "rejected_input_ids":    r_ids,
        "rejected_attention_mask": r_mask,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_val(
    model: RewardModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    chosen_scores, rejected_scores = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        r_w = model(
            batch["chosen_input_ids"].to(device),
            batch["chosen_attention_mask"].to(device),
        )
        r_l = model(
            batch["rejected_input_ids"].to(device),
            batch["rejected_attention_mask"].to(device),
        )
        chosen_scores.extend(r_w.cpu().tolist())
        rejected_scores.extend(r_l.cpu().tolist())
    return np.array(chosen_scores), np.array(rejected_scores)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

BLUE   = "#2E86AB"
RED    = "#E84855"
GRAY   = "#6B6B6B"
GREEN  = "#3BB273"
LIGHT  = "#F0F4F8"


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(LIGHT)
    ax.grid(True, color="white", linewidth=1.2, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_accuracy_curves(
    history: list[dict],
    output_path: Path,
) -> None:
    """
    Two-panel figure:
      Left  — accuracy over epochs (val, train, 50% baseline, 65% target)
      Right — Bradley-Terry loss over epochs
    """
    epochs     = [h["epoch"]    for h in history]
    val_acc    = [h["val_acc"]  for h in history]
    train_acc  = [h["train_acc"] for h in history]
    val_loss   = [h["val_loss"] for h in history]
    train_loss = [h["train_loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor("white")

    # — Accuracy panel —
    ax1.plot(epochs, val_acc,   "o-",  color=BLUE,  linewidth=2.5,  label="val accuracy",   zorder=3)
    ax1.plot(epochs, train_acc, "s--", color=BLUE,  linewidth=1.5,  alpha=0.45, label="train accuracy", zorder=3)
    ax1.axhline(0.50, color=GRAY,  linestyle=":",  linewidth=1.8, label="random baseline (50%)", zorder=2)
    ax1.axhline(0.65, color=GREEN, linestyle="--", linewidth=1.8, label="Phase 1 target (65%)", zorder=2)
    ax1.set_xlim(0.7, max(epochs) + 0.3)
    ax1.set_ylim(0.44, min(1.02, max(val_acc) + 0.08))
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Reward Model Val Accuracy", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(epochs)
    ax1.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc")
    _style_ax(ax1)

    # — Loss panel —
    ax2.plot(epochs, val_loss,   "o-",  color=RED,  linewidth=2.5, label="val loss",   zorder=3)
    ax2.plot(epochs, train_loss, "s--", color=RED,  linewidth=1.5, alpha=0.45, label="train loss", zorder=3)
    ax2.axhline(np.log(2), color=GRAY, linestyle=":", linewidth=1.8, label="random baseline (log 2 ≈ 0.693)", zorder=2)
    ax2.set_xlim(0.7, max(epochs) + 0.3)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Bradley-Terry Loss  [−log σ(rw − rl)]", fontsize=11)
    ax2.set_title("Reward Model Loss", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(epochs)
    ax2.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc")
    _style_ax(ax2)

    best_acc = max(val_acc)
    fig.suptitle(
        f"GPT-2 Bradley-Terry Reward Model on HH-RLHF  ·  best val acc = {best_acc:.1%}",
        fontsize=11, y=1.02, color=GRAY,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {output_path}")


def plot_reward_distributions(
    epoch_data: list[tuple[int, np.ndarray, np.ndarray, float]],
    output_path: Path,
) -> None:
    """
    One subplot per epoch showing chosen (blue) vs rejected (red) reward histograms.
    The distributions should visibly separate by epoch 2.
    """
    n = len(epoch_data)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.2), sharey=False)
    fig.patch.set_facecolor("white")

    if n == 1:
        axes = [axes]

    for ax, (epoch, chosen, rejected, acc) in zip(axes, epoch_data):
        lo = min(chosen.min(), rejected.min())
        hi = max(chosen.max(), rejected.max())
        pad = (hi - lo) * 0.05
        bins = np.linspace(lo - pad, hi + pad, 45)

        ax.hist(chosen,   bins=bins, alpha=0.70, color=BLUE, label="chosen",   density=True, zorder=3)
        ax.hist(rejected, bins=bins, alpha=0.70, color=RED,  label="rejected", density=True, zorder=3)

        # Means
        ax.axvline(chosen.mean(),   color=BLUE, linestyle="--", linewidth=1.4, zorder=4)
        ax.axvline(rejected.mean(), color=RED,  linestyle="--", linewidth=1.4, zorder=4)

        overlap_pct = (chosen < rejected).mean() * 100
        ax.set_xlabel("Reward score", fontsize=11)
        if epoch == epoch_data[0][0]:
            ax.set_ylabel("Density", fontsize=11)
        ax.set_title(
            f"Epoch {epoch}  ·  val acc = {acc:.1%}\n"
            f"(chosen lower than rejected in {overlap_pct:.0f}% of pairs)",
            fontsize=10, fontweight="bold", pad=8,
        )
        ax.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc")
        _style_ax(ax)

    fig.suptitle(
        "Reward Score Distributions: Chosen vs Rejected  —  GPT-2 RM, HH-RLHF val split",
        fontsize=11, y=1.02, color=GRAY,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(
    checkpoint_dir: str | Path,
    data_dir: str | Path,
    figures_dir: str | Path,
    device: torch.device,
    max_val_batches: int = 80,
) -> None:
    """Entry point callable from train_rm.py or standalone."""
    ckpt_dir    = Path(checkpoint_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    history_path = ckpt_dir / "training_history.json"
    if not history_path.exists():
        print(f"No training history at {history_path}; skipping figures.")
        return

    with open(history_path) as f:
        history = json.load(f)

    # ── Fig 1a: accuracy curves ──────────────────────────────────────────
    print("Plotting accuracy curves…")
    plot_accuracy_curves(history, figures_dir / "rm_val_accuracy.png")

    # ── Fig 1b: reward distributions per epoch ───────────────────────────
    print("Scoring val split for each epoch…")
    dataset = load_from_disk(str(data_dir))
    val_ds  = PreferenceDataset(dataset["val"])
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False,
        collate_fn=_collate, num_workers=0,  # num_workers=0 avoids fork issues when called from train
    )

    epoch_data: list[tuple] = []
    for row in history:
        epoch      = row["epoch"]
        epoch_ckpt = ckpt_dir / f"epoch_{epoch}_rm.pt"
        best_ckpt  = ckpt_dir / "best_rm.pt"

        ckpt_path = epoch_ckpt if epoch_ckpt.exists() else best_ckpt
        if not ckpt_path.exists():
            print(f"  Epoch {epoch}: no checkpoint found, skipping.")
            continue

        model  = RewardModel.from_checkpoint(str(ckpt_path)).to(device)
        chosen, rejected = score_val(model, val_loader, device, max_val_batches)
        acc    = (chosen > rejected).mean()
        epoch_data.append((epoch, chosen, rejected, float(acc)))
        print(f"  Epoch {epoch}: {len(chosen)} pairs scored  acc={acc:.4f}")
        del model  # free GPU memory before next epoch

    if epoch_data:
        plot_reward_distributions(epoch_data, figures_dir / "rm_reward_distributions.png")

    print(f"\nFig 1 saved to {figures_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fig 1 reward model figures")
    parser.add_argument("--checkpoint_dir", default="checkpoints/rm")
    parser.add_argument("--data_dir",       default="data/processed")
    parser.add_argument("--figures_dir",    default="results/figures")
    parser.add_argument("--device",         default="auto")
    parser.add_argument("--max_val_batches", type=int, default=80,
                        help="Val batches to score per epoch (80×16=1280 examples)")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    generate(args.checkpoint_dir, args.data_dir, args.figures_dir, device, args.max_val_batches)


if __name__ == "__main__":
    main()
