"""Reward-out operators (hook category = 'reward_out').

P0 (Phase 3.1):
  ReRepP  -- Reward Repeat Periodically.
  ReDistP -- Reward Disturbance Periodically: + Gaussian noise.

P1 (Phase 3.2):
  ReDisoP -- Reward Disorder Periodically: irregular modification at period
             (sign flip OR random scale, picked uniformly).

Hook signature: fn(reward, ctx, cfg) -> reward (float)
"""
from mutation.context import current_step
from mutation.registry import register
from mutation.timing import trigger


@register("ReRepP", "reward_out")
def re_rep_p(reward, ctx, cfg):
    timestep = current_step()
    if not trigger("ReRepP", timestep):
        ctx.state["last_reward"] = reward
        return reward
    last = ctx.state.get("last_reward")
    if last is None:
        ctx.state["last_reward"] = reward
        return reward
    return last


@register("ReDistP", "reward_out")
def re_dist_p(reward, ctx, cfg):
    timestep = current_step()
    if not trigger("ReDistP", timestep):
        return reward
    noise = ctx.np_rng.normal(0.0, 0.5 * cfg.intensity)
    return float(reward + noise)


@register("ReDisoP", "reward_out")
def re_diso_p(reward, ctx, cfg):
    """Period: irregular modification — 50/50 sign-flip-scaled OR random-scale."""
    timestep = current_step()
    if not trigger("ReDisoP", timestep):
        return reward
    intensity = max(float(cfg.intensity), 0.0)
    if ctx.rng.random() < 0.5:
        # intensity=0 -> original reward; intensity=1 -> full sign flip.
        return float(reward) * (1.0 - 2.0 * intensity)
    scale = 1.0 + ((ctx.rng.random() * 2.0) - 1.0) * intensity
    return float(reward) * scale
