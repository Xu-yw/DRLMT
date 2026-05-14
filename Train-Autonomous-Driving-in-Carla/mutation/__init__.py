"""Mutation runtime (Phase 2 scaffold).

Phase 3 will populate operators/ and dispatch from hooks.py.
With MUTATION_TYPE=none (default), all hooks are identity passthrough,
keeping behavior byte-identical with main.
"""
from .config import MutationConfig, get_config, reset_config
from .hooks import state_out, action_in, reward_out, rc, es_sample

__all__ = [
    "MutationConfig",
    "get_config",
    "reset_config",
    "state_out",
    "action_in",
    "reward_out",
    "rc",
    "es_sample",
]
