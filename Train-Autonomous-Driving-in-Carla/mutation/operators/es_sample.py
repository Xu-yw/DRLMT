"""Exploration-sample operators (hook category = 'es_sample').

P2 (Phase 3.3) — all 3 operators inject at ppo.py:72-73 (between dist.sample
and dist.log_prob). Caller (ppo.py) reassigns log_prob = dist.log_prob(returned)
AFTER this hook returns, so (action, log_prob) self-consistency is automatic
regardless of what the operator does (audit R1).

Hook signature: fn(action, mean, cov_mat, dist, ctx, cfg) -> action
  action: dist.sample() output, shape (1, 2) tensor
  mean: ActorCritic.actor(obs), shape (1, 2)
  cov_mat: shape (1, 2, 2) registered buffer
  dist: MultivariateNormal(mean, cov_mat) instance

Operators:
  ESRemS  -- Permanently missing exploration: replace sample with dist.mean.
  SaSDisoS -- State-Action sampling disorder: permute action dimensions.
  SaSRepP  -- State-Action sampling repeat periodically: replay last sample.
"""
import torch

from mutation.context import current_step
from mutation.registry import register
from mutation.timing import trigger


@register("ESRemS", "es_sample")
def es_rem_s(action, mean, cov_mat, dist, ctx, cfg):
    """Sustained: always return mean (no exploration)."""
    return mean.detach()


@register("SaSDisoS", "es_sample")
def sas_diso_s(action, mean, cov_mat, dist, ctx, cfg):
    """Sustained: permute action dimensions (e.g. swap steer/throttle)."""
    if action.dim() == 1:
        idx_np = ctx.np_rng.permutation(action.shape[0])
        idx = torch.from_numpy(idx_np).long().to(action.device)
        return action[idx]
    elif action.dim() == 2:
        idx_np = ctx.np_rng.permutation(action.shape[1])
        idx = torch.from_numpy(idx_np).long().to(action.device)
        return action[:, idx]
    return action


@register("SaSRepP", "es_sample")
def sas_rep_p(action, mean, cov_mat, dist, ctx, cfg):
    """Period: replay cached previous sample.

    PPO is on-policy with no replay buffer, so 'reply DSU' is reinterpreted
    as 'reuse last action sample'.
    """
    timestep = current_step()
    if not trigger("SaSRepP", timestep):
        ctx.state["last_sample"] = action.detach().clone()
        return action
    last = ctx.state.get("last_sample")
    if last is None:
        ctx.state["last_sample"] = action.detach().clone()
        return action
    return last.clone()
