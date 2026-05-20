# RLHF From Scratch

PPO-based RLHF pipeline trained from scratch on GPT-2 (124M), using Anthropic HH-RLHF preference data.

**Key result:** PPO-RLHF achieves **XX% RM win rate** vs SFT baseline at β=0.1 KL penalty *(Phase 2 in progress)*

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

## Architecture

Four models, one GPU.

```
  Prompt x
     │
     ├──────────────────────────────────────────┐
     │                                          │
     ▼                                          ▼
 Actor π_θ ──── response y ──► Reward Model    Reference π_ref
 (LoRA fine-tuned)              (GPT-2 + head)  (SFT, frozen)
     │                               │               │
     │                               │ r(x,y)        │ log π_ref(y|x)
     ▼                               └───────┬───────┘
 Critic V_φ                                  ▼
 (value baseline)              Shaped Reward = r(x,y) − β·KL(π_θ ∥ π_ref)
     │                                        │
     └────────────────────────────────────────┘
                     PPO Update
            (clipped surrogate + value loss)
```

The KL penalty at every token prevents the actor from drifting too far from the reference policy while still optimising for the reward model's preferences.

---

## Key Results

> Results from Phase 2 training — figures committed to `results/figures/` after run completes.

| Model | RM Win Rate | Avg Reward | KL from π_ref |
|-------|-------------|-----------|--------------|
| SFT baseline (π_ref) | 50% | — | 0.0 |
| PPO β=0.05 | TBD | TBD | TBD |
| PPO β=0.10 | TBD | TBD | TBD |
| PPO β=0.20 | TBD | TBD | TBD |

**Fig 2 — PPO training dashboard** *(after Phase 2)*

**Fig 3 — Reward vs KL trade-off** *(after Phase 2)*

This plot is the most important result. It reproduces the Gao et al. (2022) finding that reward and KL diverge as β decreases: small β → high reward but incoherent text (reward hacking); large β → no improvement over SFT. Our own data will show where the Pareto frontier sits for GPT-2 + HH-RLHF.

**Qualitative examples** *(after Phase 2 — will include at least one reward-hacking case)*

---

## Implementation Details

Three things that make this non-trivial:

### 1. Score from last non-padding token, not last index

```python
# Wrong — on a right-padded batch, index -1 is padding for all but the longest seq
score = hidden_states[:, -1, :]

# Correct — find the actual last real token using the attention mask
last_token_idx = attention_mask.sum(dim=1) - 1          # (B,)
batch_idx = torch.arange(B, device=hidden_states.device)
score = hidden_states[batch_idx, last_token_idx]         # (B, H)
```

See `src/reward_model.py:48`.

### 2. KL penalty prevents reward hacking

Without the KL term, PPO finds degenerate solutions fast: extremely verbose responses, repetition, and sycophantic phrasing that score well with the RM but are obviously worse to a human. The shaped reward is:

```
r_shaped(t) = r_RM(x, y) · [t == T]  −  β · log(π_θ(aₜ|x,y<t) / π_ref(aₜ|x,y<t))
```

The scalar RM score lands only on the final token; the per-token KL penalty penalises every deviation from the reference policy. The β hyperparameter in `configs/ppo_config.yaml` controls the trade-off.

### 3. Four models on one 16GB GPU

| Model | Role | Trainable | Memory trick |
|-------|------|-----------|-------------|
| Actor π_θ | Generates responses | LoRA r=8 (~0.7M) | LoRA keeps base weights frozen |
| Critic V_φ | Value baseline | LoRA r=8 | Shared backbone init with actor |
| Reward model | Scores responses | Frozen after Phase 1 | `torch.no_grad()` inference |
| Reference π_ref | KL baseline | Frozen | `torch.no_grad()` inference |

Full fine-tuning at 7B+ scale requires gradient checkpointing and CPU offloading; the configs in `configs/ppo_config.yaml` document the translation.

---

## Quickstart

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Prepare data (downloads ~100MB from HuggingFace)
make prepare-data

# 3. Train reward model  (~1–3 epochs, target val accuracy ≥ 65%)
make train-rm

# 4. Train SFT baseline  (~2 epochs)
make train-sft

# 5. PPO training  (Phase 2)
make train-ppo

# 6. Evaluate
make eval

# Run tests
make test
```

Pre-trained Phase 1 checkpoints: *(link after upload — skip to `make train-ppo` directly)*

---

## Phase Status

| Phase | Contents | Status |
|-------|----------|--------|
| **Phase 1** | Reward model + SFT baseline | ✅ implemented |
| **Phase 2** | PPO training loop | 🔜 next |
| **Phase 3** | Evaluation + ablations | 🔜 after Phase 2 |

---

## What's Intentionally Missing

**Scale**: GPT-2 (124M) instead of Llama 3 (7B+). This means the full four-model PPO setup fits on a single consumer GPU *without* LoRA on the actor, gradient checkpointing, or CPU offloading — all of which add confounding complexity. The algorithms are identical at larger scale; `configs/ppo_config.yaml` documents the hyperparameter translations.

**RLHF-specific tricks not included**: best-of-N sampling (a strong baseline that doesn't need PPO), constitutional AI self-critique, DPO (simpler alternative to PPO that skips the RL loop entirely). Each is worth understanding independently before combining.

**Reward model ensembling**: a single RM is used for clarity. Production RLHF typically uses an ensemble to reduce reward hacking.

---

## Repository Layout

```
src/
  reward_model.py       Bradley-Terry RM (Phase 1)
  sft_baseline.py       SFT + LoRA     (Phase 1)
  ppo_trainer.py        PPO loop        (Phase 2)
  rollout_buffer.py     GAE buffer      (Phase 2)
  reward_shaping.py     KL shaping      (Phase 2)
  evaluate.py           Metrics         (Phase 3)
data/
  prepare_hh_rlhf.py    Dataset prep    (Phase 1)
scripts/
  train_rm.py           RM training     (Phase 1)
  train_sft.py          SFT training    (Phase 1)
  train_ppo.py          PPO training    (Phase 2)
  run_eval.py           Evaluation      (Phase 3)
configs/
  rm_config.yaml
  sft_config.yaml
  ppo_config.yaml       includes 7B+ scaling notes
  eval_config.yaml
results/
  figures/              PNG plots committed after Phase 2
  tables/               CSV metrics
```

---

## References

- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* (InstructGPT)
- Gao et al. (2022). *Scaling Laws for Reward Model Overoptimization.* — the reward vs KL figure this repo reproduces
- Bai et al. (2022). *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.* (Anthropic HH-RLHF dataset)
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.*
