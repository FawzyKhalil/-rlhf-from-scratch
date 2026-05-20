"""
Train the Bradley-Terry reward model on HH-RLHF preference pairs.

Key decisions:
  - 1–3 epochs only: RMs overfit fast. Val accuracy plateaus or drops after ~1 epoch.
  - Cosine LR schedule with warmup: stabilises early training when rewards are near zero.
  - Save the checkpoint with highest val accuracy, not lowest val loss.
  - Log val accuracy every epoch — 65%+ is the Phase 1 exit criterion (human ≈ 75%).

Usage:
    python scripts/train_rm.py
    python scripts/train_rm.py --config configs/rm_config.yaml --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.reward_model import RewardModel, bradley_terry_loss, reward_accuracy

GPT2_PAD_ID = 50256  # eos == pad for GPT-2


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        return {
            "chosen_input_ids": torch.tensor(row["chosen_input_ids"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(row["chosen_attention_mask"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(row["rejected_input_ids"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(row["rejected_attention_mask"], dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Right-pad sequences within a batch to the same length."""

    def pad(seqs: list[torch.Tensor], pad_id: int = GPT2_PAD_ID):
        max_len = max(s.shape[0] for s in seqs)
        ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : s.shape[0]] = s
            mask[i, : s.shape[0]] = 1
        return ids, mask

    chosen_ids, chosen_mask = pad([b["chosen_input_ids"] for b in batch])
    rejected_ids, rejected_mask = pad([b["rejected_input_ids"] for b in batch])

    return {
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": chosen_mask,
        "rejected_input_ids": rejected_ids,
        "rejected_attention_mask": rejected_mask,
    }


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: RewardModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    log_interval: int = 50,
) -> tuple[float, float]:
    model.train()
    running_loss = running_acc = 0.0
    n = 0

    for step, batch in enumerate(loader):
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        r_w = model(chosen_ids, chosen_mask)
        r_l = model(rejected_ids, rejected_mask)

        loss = bradley_terry_loss(r_w, r_l)
        acc = reward_accuracy(r_w, r_l)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        running_acc += acc
        n += 1

        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / n
            avg_acc = running_acc / n
            print(f"  step {step+1:>5}/{len(loader)}  loss={avg_loss:.4f}  acc={avg_acc:.4f}")

    return running_loss / n, running_acc / n


@torch.no_grad()
def evaluate(
    model: RewardModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = total_acc = 0.0
    n = 0

    for batch in loader:
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        r_w = model(chosen_ids, chosen_mask)
        r_l = model(rejected_ids, rejected_mask)

        total_loss += bradley_terry_loss(r_w, r_l).item()
        total_acc += reward_accuracy(r_w, r_l)
        n += 1

    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument("--config", default="configs/rm_config.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--checkpoint_dir", default="checkpoints/rm")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device: {device}")

    # Data
    dataset = load_from_disk(args.data_dir)
    train_ds = PreferenceDataset(dataset["train"])
    val_ds = PreferenceDataset(dataset["val"])

    num_workers = min(4, torch.get_num_threads())
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    # Model
    model = RewardModel(
        model_name=cfg["model"]["name"],
        dropout=cfg["model"].get("dropout", 0.1),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Reward model: {n_params:.1f}M parameters")

    # Optimiser + schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    # Training
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history: list[dict] = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\n── Epoch {epoch}/{cfg['training']['epochs']} ──")
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            log_interval=cfg["training"].get("log_interval", 50),
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}  "
            f"train loss={train_loss:.4f} acc={train_acc:.4f}  "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}  "
            f"({elapsed:.0f}s)"
        )

        row = dict(
            epoch=epoch,
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_acc,
        )
        history.append(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_checkpoint(
                str(ckpt_dir / "best_rm.pt"),
                epoch=epoch, val_acc=val_acc, val_loss=val_loss, config=cfg,
            )
            print(f"  ✓ saved best checkpoint  (val_acc={val_acc:.4f})")

        # Always keep latest for resuming
        model.save_checkpoint(
            str(ckpt_dir / "latest_rm.pt"),
            epoch=epoch, val_acc=val_acc, val_loss=val_loss, config=cfg,
        )

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    if best_val_acc < 0.65:
        print(
            "⚠  val accuracy < 65% (random = 50%, human ≈ 75%). "
            "Try training for more epochs or increasing the learning rate."
        )


if __name__ == "__main__":
    main()
