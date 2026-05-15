"""State-out operators (hook category = 'state_out').

Phase 3.1 P0 operators:
  StRepP -- State Repeat Periodically: every period_steps, return cached
            previous obs (replays one step).
  StFuzS -- State Fuzz Sustained: every step, quantize obs to lower precision.

Hook signature: fn(obs, ctx, cfg) -> obs
  obs = [image_obs (np.float32 (160,80,3)), navigation_obs (np.float64 (5,))]
"""
import numpy as np

from mutation.context import current_step
from mutation.registry import register
from mutation.timing import trigger


@register("StRepP", "state_out")
def st_rep_p(obs, ctx, cfg):
    """Cache current obs on non-trigger steps; on trigger step return cached obs."""
    image_obs, navigation_obs = obs
    timestep = current_step()
    if not trigger("StRepP", timestep):
        ctx.state["last_state"] = (image_obs.copy(), navigation_obs.copy())
        return obs
    last = ctx.state.get("last_state")
    if last is None:
        ctx.state["last_state"] = (image_obs.copy(), navigation_obs.copy())
        return obs
    return [last[0].copy(), last[1].copy()]


@register("StFuzS", "state_out")
def st_fuz_s(obs, ctx, cfg):
    """Quantize obs. intensity in [0, 1] controls granularity.

    nav: bins = max(int(10 * (1 - intensity * 0.5)), 2)  -> 5..10 bins
    image: zero out low bits. bit_red = max(int(4 * intensity), 1) -> 1..4 bits
    """
    image_obs, navigation_obs = obs
    intensity = cfg.intensity
    nav_bins = max(int(10 * (1 - intensity * 0.5)), 2)
    nav_q = np.round(navigation_obs * nav_bins) / nav_bins

    bit_red = max(int(4 * intensity), 1)
    step = 2 ** bit_red
    img_q = (image_obs.astype(np.int32) // step) * step
    return [img_q.astype(image_obs.dtype), nav_q.astype(navigation_obs.dtype)]
