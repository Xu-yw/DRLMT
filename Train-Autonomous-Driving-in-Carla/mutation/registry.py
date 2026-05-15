"""Operator registration + dispatch.

Phase 3.0: no operators registered; hooks dispatchers fall back to identity.
Phase 3.1+: operators imported and decorated with @register; dispatchers route.

Hook categories (one per hook in hooks.py):
  state_out, action_in, reward_out, rc, pv_out, es_sample
"""
from collections import defaultdict


_registry = defaultdict(dict)


def register(op_name, hook_category):
    """Decorator. Registers fn under (hook_category, op_name)."""
    def _decorator(fn):
        _registry[hook_category][op_name] = fn
        return fn
    return _decorator


def get(hook_category, op_name):
    return _registry.get(hook_category, {}).get(op_name)


def list_operators(hook_category=None):
    if hook_category is not None:
        return list(_registry.get(hook_category, {}).keys())
    return {cat: list(ops.keys()) for cat, ops in _registry.items()}


def reset_registry():
    """Test helper. Wipes all registrations."""
    _registry.clear()
