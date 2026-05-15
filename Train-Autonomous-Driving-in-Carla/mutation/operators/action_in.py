"""Action-in operators (hook category = 'action_in').

P0 (Phase 3.1):
  AcRepR -- Action Repeat Randomly: replay cached previous action at random.
  AcFuzS -- Action Fuzz Sustained: quantize action every step.

P1 (Phase 3.2):
  AcDisoR -- Action Disorder Randomly: at random, replace action with a fresh
             uniform-random action in [-1, 1] (decision-making fault).

Hook signature: fn(action, ctx, cfg) -> action (np.ndarray shape (2,))
"""
import numpy as np

from mutation.context import current_step
from mutation.registry import register
from mutation.timing import trigger


@register("AcRepR", "action_in")
def ac_rep_r(action, ctx, cfg):
    timestep = current_step()
    prob = min(0.3 * cfg.intensity if cfg.intensity > 0 else 0.3, 1.0)
    if not trigger("AcRepR", timestep, rng=ctx.rng, prob=prob):
        ctx.state["last_action"] = np.asarray(action).copy()
        return action
    last = ctx.state.get("last_action")
    if last is None:
        ctx.state["last_action"] = np.asarray(action).copy()
        return action
    return last.copy()


@register("AcFuzS", "action_in")
def ac_fuz_s(action, ctx, cfg):
    action = np.asarray(action)
    bins = max(int(10 * (1 - cfg.intensity * 0.5)), 2)
    return (np.round(action * bins) / bins).astype(action.dtype)


@register("AcDisoR", "action_in")
def ac_diso_r(action, ctx, cfg):
    """Random: replace with uniform random action in [-1, 1]."""
    prob = min(0.3 * cfg.intensity if cfg.intensity > 0 else 0.3, 1.0)
    if not trigger("AcDisoR", current_step(), rng=ctx.rng, prob=prob):
        return action
    action_arr = np.asarray(action)
    new_action = np.array(
        [float(ctx.np_rng.uniform(-1.0, 1.0)),
         float(ctx.np_rng.uniform(-1.0, 1.0))],
        dtype=action_arr.dtype,
    )
    return new_action
