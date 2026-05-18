"""Operator registration + dispatch.

Phase 3.1+: bundled operators registered via @register decorators at
mutation.operators submodule import time. The bundled set is captured as
the "baseline" by _capture_baseline() once at framework init. Tests that
ad-hoc @register custom operators can call reset_registry() to restore
the baseline (i.e. drop their additions, keep bundled).

Hook categories: state_out, action_in, reward_out, rc, pv_out, es_sample.
"""

# 中文说明：本文件维护 hook_category -> op_name -> function 的二维注册表。
# 算子模块通过 @register("StRepP", "state_out") 把函数挂进表里；hooks.py 运行时按当前 hook 查表。
# 一个 mutant 可以只注册到一个 hook 类别，其他 hook 类别自然 passthrough。
# _capture_baseline() 用于保存内置 12 算子的注册状态，reset_registry() 主要给测试隔离使用。

from collections import defaultdict


_registry = defaultdict(dict)
_baseline = None  # set by _capture_baseline() after operators autoload


# 中文注释：装饰器注册入口；写在算子函数上方，import 模块时自动登记。
def register(op_name, hook_category):
    """Decorator. Registers fn under (hook_category, op_name)."""
    def _decorator(fn):
        _registry[hook_category][op_name] = fn
        return fn
    return _decorator


# 中文注释：dispatcher 用这个函数按“当前 hook 类别 + 当前 mutant 名”找具体实现。
def get(hook_category, op_name):
    return _registry.get(hook_category, {}).get(op_name)


def list_operators(hook_category=None):
    if hook_category is not None:
        return list(_registry.get(hook_category, {}).keys())
    return {cat: list(ops.keys()) for cat, ops in _registry.items()}


# 中文注释：测试隔离用；恢复内置算子表，丢弃测试临时注册的假算子。
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


# 中文注释：operators 自动加载完成后调用，记录正式内置算子集合。
def _capture_baseline():
    """Snapshot current registry as the baseline. Called once by
    mutation/__init__.py after operators autoload completes."""
    global _baseline
    _baseline = {cat: dict(ops) for cat, ops in _registry.items()}
