"""
SFT baseline: fine-tune GPT-2 on HH-RLHF chosen responses.

This checkpoint serves two roles in Phase 2:
  π_ref — frozen reference policy for KL penalty
  π_θ   — initial actor/critic weights for PPO

LoRA (r=8) reduces trainable parameters from 124M → ~0.7M, which lets
two copies of the model (actor + critic) sit comfortably on a 16GB GPU
alongside the frozen reward model and reference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel


class SFTModel(nn.Module):
    """GPT-2 fine-tuned on chosen responses with optional LoRA adapters."""

    def __init__(
        self,
        model_name: str | GPT2Config = "gpt2",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        if isinstance(model_name, GPT2Config):
            base = GPT2LMHeadModel(model_name)
        else:
            base = GPT2LMHeadModel.from_pretrained(model_name)

        self.use_lora = use_lora
        if use_lora:
            self.model = self._apply_lora(base, lora_r, lora_alpha, lora_dropout)
        else:
            self.model = base

    @staticmethod
    def _apply_lora(model: GPT2LMHeadModel, r: int, alpha: int, dropout: float):
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise ImportError("pip install peft") from e

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["c_attn", "c_proj"],
            bias="none",
        )
        return get_peft_model(model, config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.generate(input_ids, **kwargs)

    def save_pretrained(self, path: str) -> None:
        """Save to disk, merging LoRA weights back into the base model."""
        if self.use_lora:
            merged = self.model.merge_and_unload()
            merged.save_pretrained(path)
        else:
            self.model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str) -> "SFTModel":
        """Load a merged (non-LoRA) SFT checkpoint."""
        instance = cls.__new__(cls)
        super(SFTModel, instance).__init__()
        instance.model = GPT2LMHeadModel.from_pretrained(path)
        instance.use_lora = False
        return instance

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SFTTrainer:
    """Thin wrapper handling a single gradient step and eval step."""

    def __init__(
        self,
        model: SFTModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        self.model.train()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def eval_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        self.model.eval()
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device),
        )
        return outputs.loss.item()
