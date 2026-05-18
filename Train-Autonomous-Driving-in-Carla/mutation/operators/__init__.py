"""Operator implementations. Auto-imports submodules at package load to
trigger @register side effects.

Each operator module registers one or more operators via the @register
decorator from mutation.registry. The mutation framework picks them up
via mutation.registry.get(category, op_name) at dispatch time.
"""

# 中文说明：本文件的职责是自动加载 operators 目录下的所有算子模块。
# import mutation 时会执行这里的 _autoload_submodules()，逐个 import state_out/action_in/reward_out/pv_out/es_sample。
# 这些模块顶层的 @register 装饰器会立即执行，把算子函数注册到 registry。
# 因此新增算子文件后，只要放在本目录且文件名不以下划线开头，就会自动进入注册流程。

import importlib
import pkgutil


# 中文注释：扫描当前目录下的算子模块并 import；import 的副作用就是执行 @register。
def _autoload_submodules():
    for _finder, name, _ispkg in pkgutil.iter_modules(__path__):
        if name.startswith("_"):
            continue
        importlib.import_module(f".{name}", __package__)


_autoload_submodules()
