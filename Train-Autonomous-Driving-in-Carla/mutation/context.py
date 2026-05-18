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

# 中文说明：本文件保存“每个算子自己的运行上下文”。
# 很多 mutant 需要跨 step 记忆，例如 StRepP 的 last_state、StDisoP 的 history、ReRepP 的 last_reward。
# 这些状态不能全局混在一起，所以 get_context(op_name, seed) 会给每个 op 建一个独立 OperatorContext。
# reset_episode() 只清空 episode 内缓存和 global step，不重置 RNG；这样同一 run 内随机序列保持可复现。
# global step 由 action_in hook 在 env.step 入口推进，是 P/R/S timing 判断的时间基准。

import random

import numpy as np


# 中文注释：每个 op 一个 OperatorContext，负责隔离缓存、Python RNG 和 NumPy RNG。
class OperatorContext:
    def __init__(self, op_name, seed):
        self.op_name = op_name
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.state = {}

    # 中文注释：episode 级缓存必须清空，避免上一局的 last_state/history 泄漏到下一局。
    def reset_episode(self):
        self.state.clear()

    def __repr__(self):
        return f"OperatorContext(op={self.op_name!r}, seed={self.seed}, state_keys={list(self.state.keys())})"


_ctx_registry = {}
_global_step = 0


# 中文注释：同一个 op 第一次访问时创建上下文；之后即使传入不同 seed，也继续复用已有上下文。
def get_context(op_name, seed=0):
    if op_name not in _ctx_registry:
        _ctx_registry[op_name] = OperatorContext(op_name, seed)
    return _ctx_registry[op_name]


# 中文注释：global step 是 mutation timing 的计数器，由 action_in 在每个 env.step 入口推进一次。
def tick():
    global _global_step
    _global_step += 1


def current_step():
    return _global_step


# 中文注释：env.reset() 时调用；清所有算子的 episode 缓存，并把 global step 归零。
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
