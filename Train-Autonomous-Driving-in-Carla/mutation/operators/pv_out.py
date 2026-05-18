"""Policy-value output operators (hook category = 'pv_out').

P1 (Phase 3.2):
  PVDistR -- Policy-Value Disturbance Randomly: each step with prob = 0.3 * intensity,
             add Gaussian noise (std = 0.1 * intensity) to the policy output action
             tensor. log_prob IS recomputed via dist.log_prob(new_action) when dist
             is provided (training path); evaluation path (dist=None) skips recompute
             since logprob is a 0-d zero placeholder that no one reads.

Design note (per 2026-05-18 phase3.6 fix): pv_out hook now receives the
distribution, so PVDistR keeps (action, logprob) self-consistent and avoids
PPO ratio corruption. This restores the operator's intended semantics
('PV output noise' per paper Table 2) and removes the unintended
'PPO training data poisoning' side-effect.

Hook signature: fn(action, logprob, dist, ctx, cfg) -> (action, logprob)
  action: torch.Tensor shape (1, 2) on device
  logprob: torch.Tensor shape (1,) on device  (or 0-d zero if deterministic)
  dist: MultivariateNormal | None  (None when deterministic=True)
"""
import torch

from mutation.context import current_step
from mutation.registry import register
from mutation.timing import trigger


@register("PVDistR", "pv_out")
def pv_dist_r(action, logprob, dist, ctx, cfg):
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
    new_action = action + noise
    if dist is None:
        # Evaluation path: logprob is 0-d zero placeholder, no recompute needed
        return new_action, logprob
    # Training path: recompute logprob to keep (action, logprob) self-consistent
    new_logprob = dist.log_prob(new_action).detach()
    return new_action, new_logprob
