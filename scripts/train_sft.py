"""
Fine-tune GPT-2 on HH-RLHF chosen responses (SFT baseline).

The resulting checkpoint is:
  π_ref — frozen reference for KL penalty in Phase 2 PPO
  π_θ   — initial actor/critic weights for PPO

With LoRA (default), only ~0.7M of 124M parameters are trained, leaving
room for two copies of the model in GPU memory during PPO.

Usage:
    python scripts/train_sft.py
    python scripts/train_sft.py --config configs/sft_config.yaml --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sft_baseline import SFTModel, SFTTrainer

GPT2_PAD_ID = 50256


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChosenResponseDataset(Dataset):
    """Wraps the preprocessed Arrow dataset; only uses chosen_input_ids."""

    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        return {
            "input_ids": torch.tensor(row["chosen_input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["chosen_attention_mask"], dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Right-pad; labels = input_ids (causal LM loss over full sequence)."""
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = torch.full((len(batch), max_len), GPT2_PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = b["attention_mask"]
        labels[i, :n] = b["input_ids"]  # supervise every real token

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SFT baseline training")
    parser.add_argument("--config", default="configs/sft_config.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--checkpoint_dir", default="checkpoints/sft")
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

    dataset = load_from_disk(args.data_dir)
    train_ds = ChosenResponseDataset(dataset["train"])
    val_ds = ChosenResponseDataset(dataset["val"])

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

    model = SFTModel(
        model_name=cfg["model"]["name"],
        use_lora=cfg["model"].get("use_lora", True),
        lora_r=cfg["model"].get("lora_r", 8),
    ).to(device)

    trainable = model.trainable_parameters()
    total = model.total_parameters()
    print(f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M  ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    trainer = SFTTrainer(
        model, optimizer, device,
        grad_clip=cfg["training"]["max_grad_norm"],
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\n── Epoch {epoch}/{cfg['training']['epochs']} ──")
        t0 = time.time()

        # Train
        model.train()
        train_losses: list[float] = []
        log_every = cfg["training"].get("log_interval", 50)

        for step, batch in enumerate(train_loader):
            loss = trainer.train_step(batch["input_ids"], batch["attention_mask"], batch["labels"])
            scheduler.step()
            train_losses.append(loss)

            if (step + 1) % log_every == 0:
                recent = train_losses[-log_every:]
                avg = sum(recent) / len(recent)
                print(f"  step {step+1:>5}/{len(train_loader)}  loss={avg:.4f}  ppl={2**avg:.2f}")

        # Validate
        val_losses: list[float] = []
        for batch in val_loader:
            val_losses.append(
                trainer.eval_step(batch["input_ids"], batch["attention_mask"], batch["labels"])
            )

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}  "
            f"train loss={train_loss:.4f} ppl={2**train_loss:.2f}  "
            f"val loss={val_loss:.4f} ppl={2**val_loss:.2f}  "
            f"({elapsed:.0f}s)"
        )

        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(str(ckpt_dir / "best_sft"))
            # Save tokeniser alongside the model for easy loading
            GPT2Tokenizer.from_pretrained("gpt2").save_pretrained(str(ckpt_dir / "best_sft"))
            print(f"  ✓ saved best SFT checkpoint  (val_loss={val_loss:.4f})")

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}  (ppl={2**best_val_loss:.2f})")


if __name__ == "__main__":
    main()
