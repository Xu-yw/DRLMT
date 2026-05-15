"""Action-in operators (hook category = 'action_in').

Phase 3.1 P0 operators:
  AcRepR -- Action Repeat Randomly: each step, with prob = 0.3 * intensity,
            return cached previous action instead of current.
  AcFuzS -- Action Fuzz Sustained: every step, quantize action to lower precision.

Hook signature: fn(action, ctx, cfg) -> action (np.ndarray shape (2,))

Note: action_in is called after tick() in the dispatcher, so current_step()
returns the +1 value. Operators read it via ctx as needed.
"""
import numpy as np

from mutation.context import current_step
from mutation.registry import register
from mutation.timing import trigger


@register("AcRepR", "action_in")
def ac_rep_r(action, ctx, cfg):
    timestep = current_step()
    # R timing: random trigger; prob scales with intensity (clipped to [0,1])
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
    """Quantize action to bins. intensity in [0, 1]."""
    action = np.asarray(action)
    bins = max(int(10 * (1 - cfg.intensity * 0.5)), 2)
    return (np.round(action * bins) / bins).astype(action.dtype)
