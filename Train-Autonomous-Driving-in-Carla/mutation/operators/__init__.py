"""Operator implementations. Auto-imports submodules at package load to
trigger @register side effects.

Each operator module registers one or more operators via the @register
decorator from mutation.registry. The mutation framework picks them up
via mutation.registry.get(category, op_name) at dispatch time.
"""
import importlib
import pkgutil


def _autoload_submodules():
    for _finder, name, _ispkg in pkgutil.iter_modules(__path__):
        if name.startswith("_"):
            continue
        importlib.import_module(f".{name}", __package__)


_autoload_submodules()
