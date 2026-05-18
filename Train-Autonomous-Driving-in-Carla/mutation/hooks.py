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

P0-1 (2026-05-18): each dispatcher emits one [MUT-PROBE] line on its first
dispatch to a registered operator (per process lifetime). See mutation/_probe.py
for the wire format and the bug classes it guards against.
"""
from .config import get_config
from .context import get_context, tick, current_step
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
    out = fn(obs, get_context(op, cfg.seed), cfg)
    from ._probe import emit_state_out
    emit_state_out(op, cfg.intensity, obs, out, current_step())
    return out


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
    out = fn(action, get_context(op, cfg.seed), cfg)
    from ._probe import emit_action_in
    emit_action_in(op, cfg.intensity, action, out, current_step())
    return out


def reward_out(reward):
    cfg, op = _resolve()
    if op is None:
        return reward
    fn = _registry_get("reward_out", op)
    if fn is None:
        return reward
    out = fn(reward, get_context(op, cfg.seed), cfg)
    from ._probe import emit_reward_out
    emit_reward_out(op, cfg.intensity, reward, out, current_step())
    return out


def rc(coef_name, default):
    cfg, op = _resolve()
    if op is None:
        return default
    fn = _registry_get("rc", op)
    if fn is None:
        return default
    out = fn(coef_name, default, get_context(op, cfg.seed), cfg)
    from ._probe import emit_rc
    emit_rc(op, cfg.intensity, coef_name, default, out, current_step())
    return out


def pv_out(action, logprob, dist):
    """Called from agent.PPOAgent.get_action after policy returns (action, logprob, dist).

    dist: MultivariateNormal | None
        - Training path: dist is the policy distribution used to sample action
        - Evaluation path (deterministic=True): dist is None; operators must
          handle this case (either skip logprob recompute or no-op)

    Implementors that mutate action MUST keep (action, logprob) self-consistent;
    use dist.log_prob(new_action) to recompute when dist is provided.
    """
    cfg, op = _resolve()
    if op is None:
        return action, logprob
    fn = _registry_get("pv_out", op)
    if fn is None:
        return action, logprob
    new_action, new_logprob = fn(action, logprob, dist, get_context(op, cfg.seed), cfg)
    from ._probe import emit_pv_out
    emit_pv_out(op, cfg.intensity, action, new_action, logprob, new_logprob, current_step())
    return new_action, new_logprob


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
    out = fn(action, mean, cov_mat, dist, get_context(op, cfg.seed), cfg)
    from ._probe import emit_es_sample
    emit_es_sample(op, cfg.intensity, action, out, current_step())
    return out
