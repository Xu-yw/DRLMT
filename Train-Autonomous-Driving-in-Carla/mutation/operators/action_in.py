"""Action-in operators (hook category = 'action_in').

P0 (Phase 3.1):
  AcRepR -- Action Repeat Randomly: replay cached previous action at random.
  AcFuzS -- Action Fuzz Sustained: quantize action every step.

P1 (Phase 3.2):
  AcDisoR -- Action Disorder Randomly: at random, replace action with a fresh
             uniform-random action in [-1, 1] (decision-making fault).

Hook signature: fn(action, ctx, cfg) -> action (np.ndarray shape (2,))
"""

# 中文说明：本文件实现作用于环境执行动作之前的 3 个 mutant。
# action_in 位于 env.step(action) 入口，车辆 apply_control 之前；这里改的是实际会施加到 CARLA 车辆上的动作。
# 连续动作一般是两维：[steer_raw, throttle_raw]，后续代码会把 steer clip 到 [-1,1]，把第二维映射成 throttle。

import numpy as np

from mutation.context import current_step
from mutation.registry import register
from mutation.timing import trigger


# 中文注释：AcRepR，Action Repeat Randomly；随机触发时复用上一条动作。
@register("AcRepR", "action_in")
def ac_rep_r(action, ctx, cfg):
    timestep = current_step()
    # intensity 控制随机触发概率；当前实现 intensity=0 时仍 fallback 到 0.3，做 sweep 前需注意。
    prob = min(0.3 * cfg.intensity if cfg.intensity > 0 else 0.3, 1.0)
    if not trigger("AcRepR", timestep, rng=ctx.rng, prob=prob):
        ctx.state["last_action"] = np.asarray(action).copy()
        return action
    last = ctx.state.get("last_action")
    if last is None:
        ctx.state["last_action"] = np.asarray(action).copy()
        return action
    return last.copy()


# 中文注释：AcFuzS，Action Fuzz Sustained；每一步把连续动作量化到有限 bins。
@register("AcFuzS", "action_in")
def ac_fuz_s(action, ctx, cfg):
    action = np.asarray(action)
    # bins 越少动作越粗糙；最低保留 2 档避免除零或完全塌缩。
    bins = max(int(10 * (1 - cfg.intensity * 0.5)), 2)
    return (np.round(action * bins) / bins).astype(action.dtype)


# 中文注释：AcDisoR，Action Disorder Randomly；随机触发时用新的 uniform 随机动作替换策略动作。
@register("AcDisoR", "action_in")
def ac_diso_r(action, ctx, cfg):
    """Random: replace with uniform random action in [-1, 1]."""
    prob = min(0.3 * cfg.intensity if cfg.intensity > 0 else 0.3, 1.0)
    if not trigger("AcDisoR", current_step(), rng=ctx.rng, prob=prob):
        return action
    action_arr = np.asarray(action)
    # 直接在合法连续动作域 [-1, 1] 里采样两维动作。
    new_action = np.array(
        [float(ctx.np_rng.uniform(-1.0, 1.0)),
         float(ctx.np_rng.uniform(-1.0, 1.0))],
        dtype=action_arr.dtype,
    )
    return new_action
