"""Mutation hook surface (Phase 2 passthrough).

Hook injection points (anchored to runs/code_audit_phase2_20260514.md):

  state_out   - simulation/environment.py:194 (reset return) + :388 (step return)
  action_in   - simulation/environment.py around :222 (after velocity, before clip)
  reward_out  - simulation/environment.py:388 (return tuple element 2)
  rc          - simulation/environment.py:333 / 335 / 338 / 339-340
                (cruise / cruise_discrete / progress / offcent coefficients)
  es_sample   - networks/on_policy/ppo/ppo.py:72-73
                (between dist.sample and dist.log_prob; log_prob MUST be
                recomputed on the (possibly mutated) action -- see audit R1)

MUTATION_TYPE=none -> all hooks return identity. Byte-identical with main.
Phase 3 will dispatch on cfg.mutation_type to operators in mutation/operators/.
"""
from .config import get_config


def state_out(obs):
    """obs = [image_obs (np.float32 (160,80,3)), navigation_obs (np.float64 (5,))]."""
    cfg = get_config()
    if not cfg.is_active:
        return obs
    return obs


def action_in(action):
    """action = np.array shape (2,) in [-1, 1] for continuous, int for discrete."""
    cfg = get_config()
    if not cfg.is_active:
        return action
    return action


def reward_out(reward):
    """Final scalar reward leaving env.step(). Phase 3 may add noise/bias."""
    cfg = get_config()
    if not cfg.is_active:
        return reward
    return reward


def rc(coef_name, default):
    """Reward coefficient hook.

    coef_name in {'cruise', 'progress', 'offcent'}.
    default is the baseline coefficient value:
      - cruise:   1.0  (env L333, L335)
      - progress: 0.05 (env L338)
      - offcent:  0.2  (env L340, sign carried by the '-=' operator)

    Passthrough returns default unchanged. Phase 3 RC operators scale or replace.
    """
    cfg = get_config()
    if not cfg.is_active:
        return default
    return default


def es_sample(action, mean, cov_mat, dist):
    """Exploration-sample hook.

    Caller contract (ppo.py): invoke AFTER dist.sample(), BEFORE dist.log_prob(action).
    log_prob is recomputed on the possibly-mutated action so that
    (action, log_prob) stays self-consistent in memory.  Otherwise PPO ratio
    explodes at low cov (see audit R1).
    """
    cfg = get_config()
    if not cfg.is_active:
        return action
    return action
