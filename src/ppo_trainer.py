"""
PPO trainer — Phase 2.

Will implement the full clipped-surrogate PPO loop:
  - token-level KL penalty against π_ref to prevent reward hacking
  - GAE advantage estimation (λ=0.95)
  - separate actor / critic optimizers
  - PPO epochs over a fixed rollout buffer

Implemented in Phase 2 after reward model and SFT baseline checkpoints exist.
"""

raise NotImplementedError("PPO trainer is implemented in Phase 2.")
