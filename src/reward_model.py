"""
Bradley-Terry reward model built on GPT-2 (124M).

Architecture: GPT-2 backbone + scalar linear head.

Critical detail: the reward is read from the last *non-padding* token,
not blindly from index -1. On a right-padded batch, index -1 is always a
padding token for all but the longest sequence — reading from it would
make shorter sequences' rewards depend on padding, not content.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model


class RewardModel(nn.Module):
    """GPT-2 with a scalar reward head for Bradley-Terry preference learning."""

    def __init__(self, model_name: str | GPT2Config = "gpt2", dropout: float = 0.1):
        super().__init__()
        if isinstance(model_name, GPT2Config):
            self.transformer = GPT2Model(model_name)
        else:
            self.transformer = GPT2Model.from_pretrained(model_name)

        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

        # Small init keeps early-training rewards near zero and loss near log(2) ≈ 0.693
        nn.init.normal_(self.reward_head.weight, std=0.01)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, T) token ids
            attention_mask: (B, T) 1 for real tokens, 0 for padding

        Returns:
            scores: (B,) scalar reward per sequence
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, H)

        # Index of the last real token in each sequence
        if attention_mask is not None:
            last_token_idx = attention_mask.sum(dim=1) - 1  # (B,)
        else:
            last_token_idx = torch.full(
                (input_ids.shape[0],),
                input_ids.shape[1] - 1,
                device=input_ids.device,
                dtype=torch.long,
            )

        batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        last_hidden = hidden_states[batch_idx, last_token_idx]  # (B, H)
        scores = self.reward_head(self.dropout(last_hidden)).squeeze(-1)  # (B,)
        return scores

    def save_checkpoint(self, path: str, **extra) -> None:
        torch.save({"model_state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def from_checkpoint(cls, path: str, model_name: str = "gpt2") -> "RewardModel":
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt.get("config", {})
        model = cls(
            model_name=model_name,
            dropout=cfg.get("model", {}).get("dropout", 0.1),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model


def bradley_terry_loss(r_w: torch.Tensor, r_l: torch.Tensor) -> torch.Tensor:
    """
    L = -log σ(r_w - r_l)

    Minimising this pushes r_w > r_l. Equivalent to maximum-likelihood
    under a Bradley-Terry model of pairwise preferences.

    Args:
        r_w: rewards for chosen (winning) responses, shape (B,)
        r_l: rewards for rejected (losing) responses, shape (B,)

    Returns:
        scalar mean loss
    """
    return -F.logsigmoid(r_w - r_l).mean()


def reward_accuracy(r_w: torch.Tensor, r_l: torch.Tensor) -> float:
    """Fraction of pairs where the chosen response has strictly higher reward."""
    return (r_w > r_l).float().mean().item()
