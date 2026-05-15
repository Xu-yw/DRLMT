"""Policy-value output operators (hook category = 'pv_out').

P1 (Phase 3.2):
  PVDistR -- Policy-Value Disturbance Randomly: each step with prob = 0.3 * intensity,
             add Gaussian noise (std = 0.1 * intensity) to the policy output action
             tensor. log_prob is NOT recomputed.

Design note (per Phase 3 decision A): pv_out hook does not receive the
distribution; PVDistR perturbs the action and leaves log_prob aligned with
the original sample. PPO ratio clipping absorbs small mismatches; for
larger intensity this can be expected to destabilize training, which is
the operator's intended mutation effect.

Hook signature: fn(action, logprob, ctx, cfg) -> (action, logprob)
  action: torch.Tensor shape (1, 2) on device
  logprob: torch.Tensor shape (1,) on device  (or 0-d zero if deterministic)
"""
import torch

from mutation.context import current_step
from mutation.registry import register
from mutation.timing import trigger


@register("PVDistR", "pv_out")
def pv_dist_r(action, logprob, ctx, cfg):
    prob = min(0.3 * cfg.intensity if cfg.intensity > 0 else 0.3, 1.0)
    if not trigger("PVDistR", current_step(), rng=ctx.rng, prob=prob):
        return action, logprob
    noise_std = 0.1 * cfg.intensity
    noise = torch.normal(
        mean=0.0,
        std=noise_std,
        size=tuple(action.shape),
        device=action.device,
        dtype=action.dtype,
    )
    return action + noise, logprob
