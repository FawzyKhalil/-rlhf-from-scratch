"""
Reward shaping — Phase 2.

Combines the scalar RM score with a token-level KL penalty:

    shaped_reward(t) = r_RM(x, y) * [t == T]  -  β * KL(π_θ || π_ref)[t]

where T is the final token of the response. The KL term discourages the
actor from drifting too far from the reference policy at every token.

Implemented in Phase 2.
"""

raise NotImplementedError("Reward shaping is implemented in Phase 2.")
