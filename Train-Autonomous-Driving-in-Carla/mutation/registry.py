"""Operator registration + dispatch.

Phase 3.1+: bundled operators registered via @register decorators at
mutation.operators submodule import time. The bundled set is captured as
the "baseline" by _capture_baseline() once at framework init. Tests that
ad-hoc @register custom operators can call reset_registry() to restore
the baseline (i.e. drop their additions, keep bundled).

Hook categories: state_out, action_in, reward_out, rc, pv_out, es_sample.
"""
from collections import defaultdict


_registry = defaultdict(dict)
_baseline = None  # set by _capture_baseline() after operators autoload


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
    """Restore the baseline registry (drop test-only ad-hoc registrations).

    If baseline not yet captured (e.g. registry tests run before framework
    init completes), falls back to wiping the entire registry.
    """
    if _baseline is None:
        _registry.clear()
        return
    _registry.clear()
    for cat, ops in _baseline.items():
        _registry[cat].update(ops)


def _capture_baseline():
    """Snapshot current registry as the baseline. Called once by
    mutation/__init__.py after operators autoload completes."""
    global _baseline
    _baseline = {cat: dict(ops) for cat, ops in _registry.items()}
