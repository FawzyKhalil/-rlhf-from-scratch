"""
PPO-RLHF trainer — Phase 2.

Implements the full four-model loop (InstructGPT / Ouyang et al. 2022):

    Actor  π_θ   — trainable, generates responses, starts from SFT ckpt
    Critic V_φ   — trainable, predicts per-token values, starts from SFT ckpt
    Ref    π_ref — frozen SFT checkpoint, provides KL baseline
    RM     r_φ   — frozen Phase-1 checkpoint, scores responses

Memory note (GPT-2, 16 GB GPU):
  Actor + Critic ≈ 2 × 124M params (all trainable) ≈ ~500 MB each in fp32
  Ref + RM frozen → no optimizer state, small working memory in no_grad passes
  Total ≈ 3–4 GB for model weights; Adam states for actor + critic add ~2 GB
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

from src.rollout_buffer import RolloutBuffer
from src.reward_model import RewardModel
from src.reward_shaping import compute_kl_divergence, shape_rewards


# ---------------------------------------------------------------------------
# Critic model
# ---------------------------------------------------------------------------

class CriticModel(nn.Module):
    """
    Per-token value function V(s_t) where s_t = (query, response[0:t]).

    Architecture mirrors the reward model but outputs a scalar at *every*
    response token position rather than at the final token only. These per-
    token values are needed by GAE to estimate advantages.

    V(s_t) is computed from hidden_state[T_q + t − 1]:
      • position T_q − 1 is the last real query token (left-padded queries)
      • position T_q + t − 1 is after seeing t response tokens
    """

    def __init__(
        self,
        model_name: str | GPT2Config = "gpt2",
        dropout: float = 0.1,
    ):
        super().__init__()
        if isinstance(model_name, GPT2Config):
            self.transformer = GPT2Model(model_name)
        else:
            self.transformer = GPT2Model.from_pretrained(model_name)

        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.value_head.weight, std=0.01)

    def forward(
        self,
        query_ids: torch.Tensor,      # (B, T_q) — left-padded
        query_mask: torch.Tensor,     # (B, T_q)
        response_ids: torch.Tensor,   # (B, T_r) — right-padded
        response_mask: torch.Tensor,  # (B, T_r)
    ) -> torch.Tensor:                # (B, T_r) — per-token values × mask
        B, T_q = query_ids.shape
        T_r = response_ids.shape[1]

        full_ids  = torch.cat([query_ids, response_ids], dim=1)    # (B, T_q+T_r)
        full_mask = torch.cat([query_mask, response_mask], dim=1)  # (B, T_q+T_r)

        hidden = self.transformer(
            input_ids=full_ids, attention_mask=full_mask
        ).last_hidden_state  # (B, T_q+T_r, H)

        # Shifted slice: position T_q+t-1 represents state s_t = (query, response[0:t])
        # so values[:, t] = V(s_t) as required by GAE
        response_hidden = hidden[:, T_q - 1 : T_q + T_r - 1, :]  # (B, T_r, H)
        values = self.value_head(self.dropout(response_hidden)).squeeze(-1)  # (B, T_r)

        return values * response_mask

    @classmethod
    def from_sft_checkpoint(
        cls, sft_path: str, dropout: float = 0.1
    ) -> "CriticModel":
        """
        Initialise the critic backbone from the merged SFT checkpoint.

        This gives the critic the same language-understanding prior as the
        actor (they share an initialisation point but are optimised separately).
        """
        sft_lm = GPT2LMHeadModel.from_pretrained(sft_path)

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        # Borrow the transformer trunk; lm_head is discarded.
        instance.transformer = sft_lm.transformer
        hidden_size = instance.transformer.config.hidden_size
        instance.dropout = nn.Dropout(dropout)
        instance.value_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(instance.value_head.weight, std=0.01)

        return instance

    def save_checkpoint(self, path: str, **extra) -> None:
        torch.save({"model_state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def from_checkpoint(
        cls, path: str, model_name: str = "gpt2"
    ) -> "CriticModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(model_name=model_name)
        model.load_state_dict(ckpt["model_state_dict"])
        return model


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Full PPO-RLHF loop over the four models.

    Responsibilities:
      1. ``collect_rollouts`` — generate responses with actor, score with RM,
         shape rewards with KL penalty, compute critic values, fill buffer.
      2. ``ppo_step``         — run n_ppo_epochs of PPO-Clip + value loss over
         the buffer, return a metrics dict.
      3. ``save_checkpoint`` / ``load_checkpoint`` — persist actor + critic.
    """

    PAD_TOKEN_ID = 50256  # GPT-2 EOS == PAD

    def __init__(
        self,
        actor: GPT2LMHeadModel,
        critic: CriticModel,
        ref_model: GPT2LMHeadModel,   # frozen
        reward_model: RewardModel,    # frozen
        device: torch.device,
        # PPO hyper-parameters
        clip_eps: float = 0.2,
        vf_coef: float = 0.1,
        entropy_coef: float = 0.01,
        n_ppo_epochs: int = 4,
        mini_batch_size: int = 8,
        max_grad_norm: float = 1.0,
        # Reward shaping
        kl_coef: float = 0.05,
        # Generation
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.ref_model = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.device = device

        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.n_ppo_epochs = n_ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        self.kl_coef = kl_coef
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Freeze reference and reward model; they are only used for inference.
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        for p in self.reward_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()
        self.reward_model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_log_probs(
        self,
        model: nn.Module,
        query_ids: torch.Tensor,      # (B, T_q) left-padded
        query_mask: torch.Tensor,     # (B, T_q)
        response_ids: torch.Tensor,   # (B, T_r) right-padded
        response_mask: torch.Tensor,  # (B, T_r)
    ) -> torch.Tensor:                # (B, T_r) log probs × mask
        """
        Compute log π(aₜ | s≤ₜ) for each response token.

        We concatenate [query | response] and run a single forward pass.
        The logit at position T_q+t-1 predicts the token at T_q+t, so:

            log_prob[:, t] = log_softmax(logits[:, T_q+t-1, :])[response[:, t]]

        With left-padded queries, position T_q-1 is always the last real query
        token, so the first response token's log prob is well-defined.
        """
        B, T_q = query_ids.shape
        T_r = response_ids.shape[1]

        full_ids  = torch.cat([query_ids,  response_ids],  dim=1)
        full_mask = torch.cat([query_mask, response_mask], dim=1)

        outputs = model(input_ids=full_ids, attention_mask=full_mask)
        logits = outputs.logits  # (B, T_q+T_r, V)

        # Slice logits that predict each response token
        response_logits = logits[:, T_q - 1 : T_q + T_r - 1, :]  # (B, T_r, V)
        log_probs = F.log_softmax(response_logits, dim=-1)          # (B, T_r, V)

        # Gather log prob of the actual sampled token
        token_log_probs = log_probs.gather(
            dim=-1,
            index=response_ids.unsqueeze(-1),
        ).squeeze(-1)  # (B, T_r)

        return token_log_probs * response_mask

    def _actor_forward_with_entropy(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single actor forward pass returning both token log probs and mean entropy.

        We compute entropy from the full vocabulary distribution at each
        response position (not just the sampled token), so this requires one
        forward pass only.

        Returns:
            log_probs:    (B, T_r) — log π_θ(aₜ|s≤ₜ) × mask
            mean_entropy: scalar   — average over real token positions
        """
        B, T_q = query_ids.shape
        T_r = response_ids.shape[1]

        full_ids  = torch.cat([query_ids,  response_ids],  dim=1)
        full_mask = torch.cat([query_mask, response_mask], dim=1)

        outputs = self.actor(input_ids=full_ids, attention_mask=full_mask)
        logits = outputs.logits

        response_logits = logits[:, T_q - 1 : T_q + T_r - 1, :]  # (B, T_r, V)
        log_probs_dist  = F.log_softmax(response_logits, dim=-1)   # (B, T_r, V)

        # Sampled token log probs
        token_log_probs = log_probs_dist.gather(
            dim=-1, index=response_ids.unsqueeze(-1)
        ).squeeze(-1) * response_mask  # (B, T_r)

        # Entropy H = -Σ_v p_v log p_v
        probs = log_probs_dist.exp()
        entropy_per_tok = -(probs * log_probs_dist).sum(-1)  # (B, T_r)
        n_real = response_mask.sum().clamp(min=1)
        mean_entropy = (entropy_per_tok * response_mask).sum() / n_real

        return token_log_probs, mean_entropy

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_rollouts(
        self,
        prompt_ids_list: list[torch.Tensor],   # list of (T_q_i,) on CPU
        buffer: RolloutBuffer,
    ) -> dict[str, float]:
        """
        Generate responses with the actor and fill the rollout buffer.

        Steps:
          1. Left-pad prompts to a uniform length.
          2. Sample responses with actor.generate().
          3. Compute log probs under actor and ref.
          4. Score full sequences with the reward model.
          5. Shape rewards (KL penalty + RM score at final token).
          6. Compute per-token critic values.
          7. Store un-padded sequences in the buffer.

        Returns a dict of rollout statistics for logging.
        """
        B = len(prompt_ids_list)

        # --- 1. Left-pad queries ---
        max_q = max(t.shape[0] for t in prompt_ids_list)
        query_ids  = torch.full((B, max_q), self.PAD_TOKEN_ID, dtype=torch.long)
        query_mask = torch.zeros(B, max_q)
        for i, q in enumerate(prompt_ids_list):
            q_len = q.shape[0]
            query_ids[i, max_q - q_len :]  = q
            query_mask[i, max_q - q_len :]  = 1.0
        query_ids  = query_ids.to(self.device)
        query_mask = query_mask.to(self.device)

        # --- 2. Generate responses ---
        self.actor.eval()
        full_output = self.actor.generate(
            input_ids=query_ids,
            attention_mask=query_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature if self.temperature > 0 else 1.0,
            pad_token_id=self.PAD_TOKEN_ID,
            eos_token_id=self.PAD_TOKEN_ID,
        )  # (B, max_q + T_gen)

        response_ids = full_output[:, max_q:]  # (B, T_gen)
        T_gen = response_ids.shape[1]

        # Build response mask: include tokens up to and including first EOS.
        eos_cumsum    = (response_ids == self.PAD_TOKEN_ID).long().cumsum(dim=1)
        response_mask = (eos_cumsum <= 1).float()  # 1 at and before first EOS

        # --- 3. Log probs under actor and ref ---
        actor_log_probs = self._compute_log_probs(
            self.actor, query_ids, query_mask, response_ids, response_mask
        )  # (B, T_gen)

        ref_log_probs = self._compute_log_probs(
            self.ref_model, query_ids, query_mask, response_ids, response_mask
        )  # (B, T_gen)

        # --- 4. RM scores on full sequence (query + response) ---
        full_ids   = torch.cat([query_ids, response_ids], dim=1)
        full_attn  = torch.cat([query_mask, response_mask], dim=1)
        rm_scores  = self.reward_model(full_ids, full_attn)  # (B,)

        # --- 5. Shape rewards ---
        rewards = shape_rewards(
            rm_scores, actor_log_probs, ref_log_probs, response_mask, beta=self.kl_coef
        )  # (B, T_gen)

        # --- 6. Critic values ---
        self.critic.eval()
        values = self.critic(
            query_ids, query_mask, response_ids, response_mask
        )  # (B, T_gen)

        # --- 7. Store un-padded sequences ---
        kl_vals = compute_kl_divergence(actor_log_probs, ref_log_probs, response_mask)
        for i in range(B):
            real = response_mask[i].bool()
            buffer.add(
                query_ids=query_ids[i, query_mask[i].bool()],
                response_ids=response_ids[i, real],
                log_probs=actor_log_probs[i, real],
                values=values[i, real],
                rewards=rewards[i, real],
            )

        return {
            "rm_score_mean": rm_scores.mean().item(),
            "kl_mean":       kl_vals.mean().item(),
            "response_len":  response_mask.sum(dim=1).float().mean().item(),
        }

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def ppo_step(
        self,
        buffer: RolloutBuffer,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """
        Run n_ppo_epochs of PPO-Clip + value-function update over the buffer.

        Actor objective (maximised):
            L_CLIP(θ) = E[ min(r_t(θ) Â_t,  clip(r_t(θ), 1−ε, 1+ε) Â_t) ]

        Critic objective (minimised):
            L_VF(φ)   = E[ max( (V_φ − R_t)², (V_clip − R_t)² ) ]   [clipped]

        Total loss (minimised):
            L = −L_CLIP  +  vf_coef · L_VF  −  entropy_coef · H

        Returns aggregated metrics averaged over all inner epochs and batches.
        """
        self.actor.train()
        self.critic.train()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for _ in range(self.n_ppo_epochs):
            for batch in buffer.mini_batches(self.mini_batch_size, self.device):
                q_ids     = batch["query_ids"]
                q_mask    = batch["query_mask"]
                r_ids     = batch["response_ids"]
                r_mask    = batch["response_mask"]
                old_lp    = batch["old_log_probs"]   # (B, T_r)
                old_vals  = batch["old_values"]       # (B, T_r)
                adv       = batch["advantages"]       # (B, T_r)
                returns   = batch["returns"]          # (B, T_r)

                # --- Actor forward (with grad) ---
                new_log_probs, entropy = self._actor_forward_with_entropy(
                    q_ids, q_mask, r_ids, r_mask
                )

                # --- Critic forward (with grad) ---
                new_values = self.critic(q_ids, q_mask, r_ids, r_mask)  # (B, T_r)

                # --- PPO-Clip loss ---
                # ratio r_t(θ) = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)
                log_ratio = new_log_probs - old_lp              # (B, T_r)
                ratio     = log_ratio.exp()                     # (B, T_r)

                surr1 = ratio * adv
                surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                # Mean over real tokens only
                n_real = r_mask.sum().clamp(min=1)
                actor_loss = -(torch.min(surr1, surr2) * r_mask).sum() / n_real

                # Approximate KL for early stopping diagnostics
                approx_kl = ((old_lp - new_log_probs) * r_mask).sum() / n_real

                # --- Clipped value loss ---
                # Clipping the value update prevents large value jumps that
                # destabilise the advantage estimates in the next rollout.
                v_clipped   = old_vals + (new_values - old_vals).clamp(
                    -self.clip_eps, self.clip_eps
                )
                vf_loss1    = (new_values - returns).pow(2)
                vf_loss2    = (v_clipped  - returns).pow(2)
                critic_loss = (torch.max(vf_loss1, vf_loss2) * r_mask).sum() / n_real

                # --- Total loss ---
                loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                actor_optimizer.step()
                critic_optimizer.step()

                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy     += entropy.item()
                total_approx_kl   += approx_kl.item()
                n_updates         += 1

        denom = max(n_updates, 1)
        return {
            "actor_loss":  total_actor_loss  / denom,
            "critic_loss": total_critic_loss / denom,
            "entropy":     total_entropy     / denom,
            "approx_kl":   total_approx_kl   / denom,
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, directory: str, step: int) -> None:
        import os
        os.makedirs(directory, exist_ok=True)
        torch.save(
            {"model_state_dict": self.actor.state_dict(), "step": step},
            os.path.join(directory, f"actor_step{step}.pt"),
        )
        torch.save(
            {"model_state_dict": self.critic.state_dict(), "step": step},
            os.path.join(directory, f"critic_step{step}.pt"),
        )

    def load_checkpoint(self, actor_path: str, critic_path: str) -> None:
        actor_ckpt  = torch.load(actor_path,  map_location="cpu", weights_only=False)
        critic_ckpt = torch.load(critic_path, map_location="cpu", weights_only=False)
        self.actor.load_state_dict(actor_ckpt["model_state_dict"])
        self.critic.load_state_dict(critic_ckpt["model_state_dict"])
