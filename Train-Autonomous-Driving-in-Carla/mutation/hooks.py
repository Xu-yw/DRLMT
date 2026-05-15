"""Mutation hook surface (Phase 3.0 dispatcher).

Hook injection points (anchored to runs/code_audit_phase2_20260514.md):

  state_out   - simulation/environment.py:194 (reset return) + :388 (step return)
  action_in   - simulation/environment.py around :222 (after velocity, before clip);
                also advances the global step counter via context.tick()
  reward_out  - simulation/environment.py:388 (return tuple element 2)
  rc          - simulation/environment.py:333 / 335 / 338 / 339-340
                (cruise / cruise_discrete / progress / offcent coefficients)
  pv_out      - agent.py:62 (after get_action_and_log_prob, before memory push)
  es_sample   - networks/on_policy/ppo/ppo.py:72-73
                (between dist.sample and dist.log_prob; log_prob MUST be
                recomputed on the (possibly mutated) action -- see audit R1)

Dispatcher contract:
  - MUTATION_TYPE=none -> all hooks return identity (early exit)
  - MUTATION_TYPE=<op> and registry.get(category, op) is None -> identity fallback
  - MUTATION_TYPE=<op> and registered -> route to operator(payload, ctx, cfg)
"""
from .config import get_config
from .context import get_context, tick
from .registry import get as _registry_get


def _resolve():
    cfg = get_config()
    if not cfg.is_active:
        return None, None
    return cfg, cfg.mutation_type


def state_out(obs):
    """obs = [image_obs (np.float32 (160,80,3)), navigation_obs (np.float64 (5,))]."""
    cfg, op = _resolve()
    if op is None:
        return obs
    fn = _registry_get("state_out", op)
    if fn is None:
        return obs
    return fn(obs, get_context(op, cfg.seed), cfg)


def action_in(action):
    """action = np.array shape (2,) for continuous, int for discrete.

    Side-effect: advances the global step counter (one tick per env.step entry).
    """
    tick()
    cfg, op = _resolve()
    if op is None:
        return action
    fn = _registry_get("action_in", op)
    if fn is None:
        return action
    return fn(action, get_context(op, cfg.seed), cfg)


def reward_out(reward):
    cfg, op = _resolve()
    if op is None:
        return reward
    fn = _registry_get("reward_out", op)
    if fn is None:
        return reward
    return fn(reward, get_context(op, cfg.seed), cfg)


def rc(coef_name, default):
    cfg, op = _resolve()
    if op is None:
        return default
    fn = _registry_get("rc", op)
    if fn is None:
        return default
    return fn(coef_name, default, get_context(op, cfg.seed), cfg)


def pv_out(action, logprob):
    """Called from agent.PPOAgent.get_action after policy returns (action, logprob).

    Implementors must keep (action, logprob) self-consistent; if action is mutated,
    logprob should be either recomputed or left coherent for PPO ratio stability.
    """
    cfg, op = _resolve()
    if op is None:
        return action, logprob
    fn = _registry_get("pv_out", op)
    if fn is None:
        return action, logprob
    return fn(action, logprob, get_context(op, cfg.seed), cfg)


def es_sample(action, mean, cov_mat, dist):
    """Caller contract (ppo.py): invoke AFTER dist.sample(), BEFORE dist.log_prob(action).

    log_prob is recomputed on the possibly-mutated action so that
    (action, log_prob) stays self-consistent in memory (audit R1).
    """
    cfg, op = _resolve()
    if op is None:
        return action
    fn = _registry_get("es_sample", op)
    if fn is None:
        return action
    return fn(action, mean, cov_mat, dist, get_context(op, cfg.seed), cfg)
