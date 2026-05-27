"""
Reward shaping for PPO-RLHF (InstructGPT / Stiennon et al. 2020).

The shaped per-token reward is:

    r_total(x, y)_t = r_RM(x, y) · 𝟙[t == T]  −  β · KL_t(π_θ ∥ π_ref)

where:
    KL_t(π_θ ∥ π_ref) = log π_θ(aₜ | s≤ₜ) − log π_ref(aₜ | s≤ₜ)

The RM score lands at the *last real token* of the response; the per-token KL
penalty discourages the actor from drifting off-distribution at every step.
β = 0.05 is a sensible starting point — increase if KL explodes, decrease if
the policy makes no progress.

All functions operate on masked tensors (response_mask zeroes out padding).
"""

from __future__ import annotations

import torch


def compute_kl_penalty(
    actor_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Sample-based per-token KL: KL_t = log π_θ(aₜ) − log π_ref(aₜ).

    This is an *unbiased single-sample estimate* of KL, not the full
    distributional KL over the vocabulary. It matches the InstructGPT
    objective and is what we use for reward shaping.

    Individual tokens can have KL < 0 (if the actor assigns lower log prob
    than ref); the sum over a full sequence is always ≥ 0 in expectation.

    Args:
        actor_log_probs: (B, T_r) — log π_θ(aₜ|s≤ₜ) for each response token
        ref_log_probs:   (B, T_r) — log π_ref(aₜ|s≤ₜ) for each response token

    Returns:
        kl: (B, T_r) — per-token KL estimate
    """
    return actor_log_probs - ref_log_probs


def shape_rewards(
    rm_scores: torch.Tensor,
    actor_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """
    Apply the InstructGPT KL penalty to the raw RM scores.

        shaped_reward_t = −β · KL_t  +  r_RM · 𝟙[t == T]

    The RM score is a scalar per sequence; it is added to the reward of the
    *last real response token* (T = last index where response_mask == 1).

    Padding positions are zeroed out by multiplying with response_mask.

    Args:
        rm_scores:        (B,)    — scalar reward model score per sequence
        actor_log_probs:  (B, T_r)— log probs under the current actor
        ref_log_probs:    (B, T_r)— log probs under the frozen reference
        response_mask:    (B, T_r)— 1 for real tokens, 0 for padding
        beta:             KL coefficient (start at 0.05; tune after inspecting
                          the KL budget in training logs)

    Returns:
        rewards: (B, T_r) — shaped per-token rewards, zero at padding positions
    """
    B = rm_scores.shape[0]

    # Per-token KL penalty (negative because KL is a cost)
    kl = compute_kl_penalty(actor_log_probs, ref_log_probs)  # (B, T_r)
    rewards = -beta * kl                                      # (B, T_r)

    # Add scalar RM score at the last real token of each sequence
    last_idx = response_mask.sum(dim=1).long() - 1   # (B,)  index of last real token
    last_idx = last_idx.clamp(min=0)                 # guard against empty sequences
    batch_idx = torch.arange(B, device=rm_scores.device)
    rewards[batch_idx, last_idx] = rewards[batch_idx, last_idx] + rm_scores

    # Zero out padding positions
    return rewards * response_mask


def compute_kl_divergence(
    actor_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sequence total KL divergence for logging / monitoring.

    KL(π_θ ∥ π_ref) = Σ_t KL_t   (summed over real response tokens)

    A climbing mean KL with a stagnant reward signal is a sign of reward
    hacking. The KL budget β controls the trade-off; see Gao et al. 2022.

    Args:
        actor_log_probs: (B, T_r)
        ref_log_probs:   (B, T_r)
        response_mask:   (B, T_r)

    Returns:
        kl_per_seq: (B,) — total KL (bits) per sequence
    """
    kl_per_token = compute_kl_penalty(actor_log_probs, ref_log_probs)
    return (kl_per_token * response_mask).sum(dim=1)
