"""
Rollout buffer for PPO.

Stores one batch of experience collected from the current actor policy:
  (query_ids, response_ids, log_probs, values, rewards, advantages, returns)

Sequences are stored un-padded (real tokens only) and padded on-the-fly
when mini-batches are yielded. Queries are left-padded; responses right-padded.
This matches the generation convention where queries are left-padded so the
response always starts immediately after the last real query token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch


@dataclass
class RolloutBuffer:
    """
    Stores one PPO rollout batch.

    All tensors are stored on CPU (variable length, no padding) to avoid
    fragmenting GPU memory between rollout collection and the PPO update.
    Padding is applied lazily inside :meth:`mini_batches`.
    """

    # Each list entry is a 1-D tensor of real (non-padding) tokens/values.
    query_ids: list[torch.Tensor] = field(default_factory=list)    # (T_q,)
    response_ids: list[torch.Tensor] = field(default_factory=list) # (T_r,)
    log_probs: list[torch.Tensor] = field(default_factory=list)    # (T_r,)
    values: list[torch.Tensor] = field(default_factory=list)       # (T_r,)
    rewards: list[torch.Tensor] = field(default_factory=list)      # (T_r,)

    # Set by compute_advantages(); one tensor per sequence.
    advantages: list[torch.Tensor] | None = None  # each (T_r,)
    returns: list[torch.Tensor] | None = None     # each (T_r,)

    def add(
        self,
        query_ids: torch.Tensor,    # (T_q,) — real tokens, no padding
        response_ids: torch.Tensor, # (T_r,) — real tokens, no padding
        log_probs: torch.Tensor,    # (T_r,)
        values: torch.Tensor,       # (T_r,)
        rewards: torch.Tensor,      # (T_r,)
    ) -> None:
        """Append a single sequence of experience (CPU-stored)."""
        assert query_ids.dim() == 1, "Pass un-padded 1-D query tensors."
        assert response_ids.dim() == 1, "Pass un-padded 1-D response tensors."
        self.query_ids.append(query_ids.detach().cpu())
        self.response_ids.append(response_ids.detach().cpu())
        self.log_probs.append(log_probs.detach().cpu())
        self.values.append(values.detach().cpu())
        self.rewards.append(rewards.detach().cpu())

    def __len__(self) -> int:
        return len(self.query_ids)

    def compute_advantages(self, gamma: float = 1.0, lam: float = 0.95) -> None:
        """
        Compute GAE-λ advantages and λ-returns in-place.

        GAE (Schulman et al. 2015):
            δ_t  = r_t + γ V_{t+1} − V_t          (TD residual)
            A_t  = Σ_{k≥0} (γλ)^k δ_{t+k}         (exponentially-weighted)

        Returns (critic targets):
            R_t = A_t + V_t

        After the loop, advantages are *whitened* (mean 0, std 1) across the
        whole buffer so that the PPO clipping threshold is scale-invariant.

        Args:
            gamma: discount factor (1.0 for language — no time preference)
            lam:   GAE-λ (0 → TD(0), 1 → full MC; 0.95 is a standard default)
        """
        advantages_list: list[torch.Tensor] = []
        returns_list: list[torch.Tensor] = []

        for rewards_i, values_i in zip(self.rewards, self.values):
            T = len(rewards_i)
            advantages_i = torch.zeros(T)
            gae = 0.0

            # Backward sweep through token positions.
            for t in reversed(range(T)):
                next_val = values_i[t + 1].item() if t + 1 < T else 0.0
                delta = rewards_i[t].item() + gamma * next_val - values_i[t].item()
                gae = delta + gamma * lam * gae
                advantages_i[t] = gae

            returns_i = advantages_i + values_i
            advantages_list.append(advantages_i)
            returns_list.append(returns_i)

        # Whiten advantages across the entire buffer for training stability.
        all_adv = torch.cat(advantages_list)
        adv_mean = all_adv.mean()
        adv_std = all_adv.std().clamp(min=1e-8)
        self.advantages = [(a - adv_mean) / adv_std for a in advantages_list]
        self.returns = returns_list

    def mini_batches(
        self,
        mini_batch_size: int,
        device: torch.device,
        pad_token_id: int = 50256,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """
        Yield randomly-shuffled mini-batches as padded tensors on ``device``.

        Yielded dict keys:
            query_ids      (B, T_q)  — left-padded
            query_mask     (B, T_q)  — 1 for real tokens
            response_ids   (B, T_r)  — right-padded
            response_mask  (B, T_r)  — 1 for real tokens
            old_log_probs  (B, T_r)
            old_values     (B, T_r)
            advantages     (B, T_r)
            returns        (B, T_r)
        """
        assert self.advantages is not None, (
            "Call compute_advantages() before iterating mini-batches."
        )

        n = len(self)
        indices = torch.randperm(n).tolist()

        for start in range(0, n, mini_batch_size):
            idx = indices[start : start + mini_batch_size]
            B = len(idx)

            q_list  = [self.query_ids[i]    for i in idx]
            r_list  = [self.response_ids[i] for i in idx]
            lp_list = [self.log_probs[i]    for i in idx]
            v_list  = [self.values[i]       for i in idx]
            adv_list = [self.advantages[i]  for i in idx]
            ret_list = [self.returns[i]     for i in idx]

            max_q = max(q.shape[0] for q in q_list)
            max_r = max(r.shape[0] for r in r_list)

            q_pad   = torch.full((B, max_q), pad_token_id, dtype=torch.long)
            q_mask  = torch.zeros(B, max_q)
            r_pad   = torch.full((B, max_r), pad_token_id, dtype=torch.long)
            r_mask  = torch.zeros(B, max_r)
            lp_pad  = torch.zeros(B, max_r)
            v_pad   = torch.zeros(B, max_r)
            adv_pad = torch.zeros(B, max_r)
            ret_pad = torch.zeros(B, max_r)

            for j in range(B):
                q_len = q_list[j].shape[0]
                r_len = r_list[j].shape[0]

                # Left-pad queries (real tokens flush-right)
                q_pad[j, max_q - q_len :]   = q_list[j]
                q_mask[j, max_q - q_len :]  = 1.0

                # Right-pad responses
                r_pad[j, :r_len]   = r_list[j]
                r_mask[j, :r_len]  = 1.0
                lp_pad[j, :r_len]  = lp_list[j]
                v_pad[j, :r_len]   = v_list[j]
                adv_pad[j, :r_len] = adv_list[j]
                ret_pad[j, :r_len] = ret_list[j]

            yield {
                "query_ids":     q_pad.to(device),
                "query_mask":    q_mask.to(device),
                "response_ids":  r_pad.to(device),
                "response_mask": r_mask.to(device),
                "old_log_probs": lp_pad.to(device),
                "old_values":    v_pad.to(device),
                "advantages":    adv_pad.to(device),
                "returns":       ret_pad.to(device),
            }

    def clear(self) -> None:
        """Reset the buffer for the next rollout collection."""
        self.query_ids.clear()
        self.response_ids.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.advantages = None
        self.returns = None
