"""Per-operator context: isolated rng + free-form state dict + episode lifecycle.

Lifecycle:
  - get_context(op_name) returns (and lazily creates) a context with a seeded rng
  - tick()  advances the module-global step counter; called by action_in hook
  - current_step() returns it
  - reset_episode() clears in-episode caches (last_state, history) for every op
                    and resets the global step counter to 0
  - reset_all() wipes the entire registry (test helper)

rng is preserved across reset_episode() so that randomized operators stay
seed-deterministic across episodes within a single training run.
"""
import random

import numpy as np


class OperatorContext:
    def __init__(self, op_name, seed):
        self.op_name = op_name
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.state = {}

    def reset_episode(self):
        self.state.clear()

    def __repr__(self):
        return f"OperatorContext(op={self.op_name!r}, seed={self.seed}, state_keys={list(self.state.keys())})"


_ctx_registry = {}
_global_step = 0


def get_context(op_name, seed=0):
    if op_name not in _ctx_registry:
        _ctx_registry[op_name] = OperatorContext(op_name, seed)
    return _ctx_registry[op_name]


def tick():
    global _global_step
    _global_step += 1


def current_step():
    return _global_step


def reset_episode():
    for ctx in _ctx_registry.values():
        ctx.reset_episode()
    global _global_step
    _global_step = 0


def reset_all():
    """Test helper: wipe context registry + counter."""
    global _global_step
    _ctx_registry.clear()
    _global_step = 0
