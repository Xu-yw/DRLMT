"""Mutation configuration. Loaded once from env vars; module-level singleton.

Env vars:
  MUTATION_TYPE       - operator name (e.g. StRepP, ESRemS); 'none' disables all hooks
  MUTATION_SEED       - int seed for mutation rng (independent of training/eval seeds)
  MUTATION_INTENSITY  - float intensity parameter, operator-specific meaning

Phase 2: only MUTATION_TYPE=none is wired (passthrough). Phase 3 will dispatch.
"""

# 中文说明：本文件负责把外部运行配置转换成进程内的 MutationConfig。
# 变异训练不是通过改代码选择算子，而是通过环境变量选择：
#   MUTATION_TYPE      选择当前 mutant；none 表示完全关闭变异。
#   MUTATION_SEED      给变异算子的随机源用，独立于 PPO/CARLA 其它随机种子。
#   MUTATION_INTENSITY 控制扰动强度，具体含义由每个算子解释。
# get_config() 是懒加载 singleton：一个 Python 进程内第一次读取 env 后会缓存。
# 因此 wrapper 在启动新 mutant 进程时传 env；测试里用 reset_config() 强制重新读取。

import os
import random

import numpy as np


# 中文注释：MutationConfig 是本进程当前 mutant 的配置快照；算子运行时只读它，不直接读 os.environ。
class MutationConfig:
    def __init__(self, mutation_type="none", seed=0, intensity=1.0):
        self.mutation_type = mutation_type
        self.seed = int(seed)
        self.intensity = float(intensity)
        self._rng = None
        self._np_rng = None
        if self.is_active:
            self._rng = random.Random(self.seed)
            self._np_rng = np.random.default_rng(self.seed)

    # 中文注释：从环境变量构造配置；wrapper 每启动一个 mutant 进程就通过 env 注入这些值。
    @classmethod
    def from_env(cls):
        return cls(
            mutation_type=os.environ.get("MUTATION_TYPE", "none"),
            seed=os.environ.get("MUTATION_SEED", "0"),
            intensity=os.environ.get("MUTATION_INTENSITY", "1.0"),
        )

    @property
    def is_active(self):
        return self.mutation_type != "none"

    @property
    def rng(self):
        return self._rng

    @property
    def np_rng(self):
        return self._np_rng

    def __repr__(self):
        return (
            f"MutationConfig(type={self.mutation_type!r}, "
            f"seed={self.seed}, intensity={self.intensity})"
        )


_config_singleton = None


# 中文注释：懒加载 singleton；避免每个 step 都重复读环境变量，也保证同一进程内配置稳定。
def get_config():
    """Return module-level MutationConfig singleton (lazy-init from env)."""
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = MutationConfig.from_env()
    return _config_singleton


# 中文注释：测试辅助函数；生产训练通常依赖新进程读取新 env，不在 step 中频繁 reset。
def reset_config(cfg=None):
    """Test helper. cfg=None forces re-read from env on next get_config()."""
    global _config_singleton
    _config_singleton = cfg


# 中文注释：训练/评估入口可调用 init()，确保 env 刚设置后 singleton 被刷新。
def init():
    """Entry-point hook: force reload from env vars. Called by training/eval
    scripts after setting MUTATION_TYPE/MUTATION_SEED/MUTATION_INTENSITY env vars
    so the singleton picks up the current process environment."""
    reset_config(None)
    return get_config()
