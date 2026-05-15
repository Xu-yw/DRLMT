"""Reward-out operators (hook category = 'reward_out').

Phase 3.1 P0 operators:
  ReRepP  -- Reward Repeat Periodically: every period_steps, replay cached
             previous reward.
  ReDistP -- Reward Disturbance Periodically: every period_steps, add Gaussian
             noise with std = 0.5 * intensity.

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
