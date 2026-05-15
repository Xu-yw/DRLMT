"""State-out operators (hook category = 'state_out').

P0 (Phase 3.1):
  StRepP -- State Repeat Periodically: replay cached previous obs at period.
  StFuzS -- State Fuzz Sustained: quantize obs every step.

P1 (Phase 3.2):
  StDistP -- State Disturbance Periodically: add Gaussian noise at period.
  StDisoP -- State Disorder Periodically: replace obs with random pick from
             rolling history at period.

Hook signature: fn(obs, ctx, cfg) -> obs
  obs = [image_obs (np.float32 (160,80,3)), navigation_obs (np.float64 (5,))]
"""
import numpy as np

from mutation.context import current_step
from mutation.registry import register
from mutation.timing import trigger


@register("StRepP", "state_out")
def st_rep_p(obs, ctx, cfg):
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
    image_obs, navigation_obs = obs
    intensity = cfg.intensity
    nav_bins = max(int(10 * (1 - intensity * 0.5)), 2)
    nav_q = np.round(navigation_obs * nav_bins) / nav_bins

    bit_red = max(int(4 * intensity), 1)
    step = 2 ** bit_red
    img_q = (image_obs.astype(np.int32) // step) * step
    return [img_q.astype(image_obs.dtype), nav_q.astype(navigation_obs.dtype)]


@register("StDistP", "state_out")
def st_dist_p(obs, ctx, cfg):
    """Period: add Gaussian noise. image std = 0.05*intensity, nav std = 0.1*intensity."""
    if not trigger("StDistP", current_step()):
        return obs
    image_obs, navigation_obs = obs
    nav_noise = ctx.np_rng.normal(0.0, 0.1 * cfg.intensity, navigation_obs.shape)
    img_noise = ctx.np_rng.normal(0.0, 0.05 * cfg.intensity, image_obs.shape)
    return [
        (image_obs + img_noise).astype(image_obs.dtype),
        (navigation_obs + nav_noise).astype(navigation_obs.dtype),
    ]


@register("StDisoP", "state_out")
def st_diso_p(obs, ctx, cfg):
    """Period: replace current obs with a random pick from rolling history (capacity 100)."""
    timestep = current_step()
    history = ctx.state.setdefault("history", [])
    if not trigger("StDisoP", timestep):
        history.append((obs[0].copy(), obs[1].copy()))
        if len(history) > 100:
            history.pop(0)
        return obs
    if len(history) < 5:
        history.append((obs[0].copy(), obs[1].copy()))
        return obs
    pick = int(ctx.np_rng.integers(0, len(history)))
    img_pick, nav_pick = history[pick]
    history.append((obs[0].copy(), obs[1].copy()))
    if len(history) > 100:
        history.pop(0)
    return [img_pick.copy(), nav_pick.copy()]
