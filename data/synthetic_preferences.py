"""
Synthetic preference generation — Phase 3.

Generates (prompt, chosen, rejected) pairs by sampling K candidate responses
from the PPO actor for each prompt and using the reward model to rank them.
The highest-scored response becomes "chosen"; the lowest becomes "rejected".

This enables ablations on:
  - Data quality: vary K (more candidates → higher-quality chosen/rejected pairs)
  - Dataset size: vary n_prompts passed to this function
"""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.reward_model import RewardModel

PAD_TOKEN_ID = 50256


@torch.no_grad()
def generate_synthetic_preferences(
    actor: GPT2LMHeadModel,
    reward_model: RewardModel,
    tokenizer: GPT2Tokenizer,
    prompts: list[str],
    device: torch.device,
    K: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    batch_size: int = 8,
) -> list[dict]:
    """
    For each prompt sample K responses from actor, score with RM, return best/worst pair.

    Args:
        prompts:    Raw text prompts.
        K:          Candidates per prompt. Higher K → bigger chosen/rejected score gap.
        batch_size: Prompts processed together (affects GPU memory, not output quality).

    Returns:
        List of dicts:  prompt, chosen_response, rejected_response,
                        chosen_score, rejected_score
    """
    actor.eval().to(device)
    reward_model.eval().to(device)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    records: list[dict] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        B = len(batch_prompts)

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        query_ids  = enc["input_ids"].to(device)       # (B, T_q)
        query_mask = enc["attention_mask"].to(device)  # (B, T_q)
        T_q = query_ids.shape[1]

        # Repeat each query K times so we can generate K responses per prompt in one pass.
        query_ids_rep  = query_ids.repeat_interleave(K, dim=0)   # (B*K, T_q)
        query_mask_rep = query_mask.repeat_interleave(K, dim=0)  # (B*K, T_q)

        full_output = actor.generate(
            input_ids=query_ids_rep,
            attention_mask=query_mask_rep,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=PAD_TOKEN_ID,
        )  # (B*K, T_q + T_gen)

        response_ids = full_output[:, T_q:]  # (B*K, T_gen)

        # Build attention mask including first EOS token (same convention as PPO trainer).
        eos_cumsum    = (response_ids == PAD_TOKEN_ID).long().cumsum(dim=1)
        response_mask = (eos_cumsum <= 1).float()

        full_ids  = torch.cat([query_ids_rep, response_ids],  dim=1)   # (B*K, T_q+T_gen)
        full_mask = torch.cat([query_mask_rep, response_mask], dim=1)  # (B*K, T_q+T_gen)
        rm_scores = reward_model(full_ids, full_mask)  # (B*K,)

        rm_scores_bk    = rm_scores.view(B, K)         # (B, K)
        response_ids_bk = response_ids.view(B, K, -1)  # (B, K, T_gen)

        best_k  = rm_scores_bk.argmax(dim=1)  # (B,)
        worst_k = rm_scores_bk.argmin(dim=1)  # (B,)

        for i in range(B):
            chosen_ids   = response_ids_bk[i, best_k[i]]
            rejected_ids = response_ids_bk[i, worst_k[i]]

            # Strip PAD (EOS) tokens for clean text decoding.
            chosen_tokens   = chosen_ids[chosen_ids != PAD_TOKEN_ID].tolist()
            rejected_tokens = rejected_ids[rejected_ids != PAD_TOKEN_ID].tolist()

            records.append({
                "prompt":            batch_prompts[i],
                "chosen_response":   tokenizer.decode(chosen_tokens,   skip_special_tokens=True),
                "rejected_response": tokenizer.decode(rejected_tokens, skip_special_tokens=True),
                "chosen_score":      rm_scores_bk[i, best_k[i]].item(),
                "rejected_score":    rm_scores_bk[i, worst_k[i]].item(),
            })

    return records


def save_synthetic_dataset(records: list[dict], output_path: str) -> None:
    """Persist synthetic preference records as a HuggingFace dataset on disk."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).save_to_disk(output_path)
    print(f"Saved {len(records)} synthetic preference pairs to {output_path}")
