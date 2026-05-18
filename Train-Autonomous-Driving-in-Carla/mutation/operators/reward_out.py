"""Reward-out operators (hook category = 'reward_out').

P0 (Phase 3.1):
  ReRepP  -- Reward Repeat Periodically.
  ReDistP -- Reward Disturbance Periodically: + Gaussian noise.

P1 (Phase 3.2):
  ReDisoP -- Reward Disorder Periodically: irregular modification at period
             (sign flip OR random scale, picked uniformly).

Hook signature: fn(reward, ctx, cfg) -> reward (float)
"""

# 中文说明：本文件实现作用于最终 reward channel 的 3 个 mutant。
# reward_out 位于 environment.step() 返回前：done、done_reason、环境内部奖励公式已经算完，
# 这里改的是 PPO memory 最终收到的 reward 值。它适合模拟奖励信号重复、加噪或错乱。

from mutation.context import current_step
from mutation.registry import register
from mutation.timing import trigger


# 中文注释：ReRepP，Reward Repeat Periodically；周期触发时复用上一份 reward。
@register("ReRepP", "reward_out")
def re_rep_p(reward, ctx, cfg):
    timestep = current_step()
    # 未触发时记录 last_reward；触发时返回缓存，模拟 reward 信号滞后。
    if not trigger("ReRepP", timestep):
        ctx.state["last_reward"] = reward
        return reward
    last = ctx.state.get("last_reward")
    if last is None:
        ctx.state["last_reward"] = reward
        return reward
    return last


# 中文注释：ReDistP，Reward Disturbance Periodically；周期触发时给 reward 加高斯噪声。
@register("ReDistP", "reward_out")
def re_dist_p(reward, ctx, cfg):
    timestep = current_step()
    if not trigger("ReDistP", timestep):
        return reward
    # intensity 线性控制 reward 噪声标准差。
    noise = ctx.np_rng.normal(0.0, 0.5 * cfg.intensity)
    return float(reward + noise)


# 中文注释：ReDisoP，Reward Disorder Periodically；周期触发时随机做符号翻转缩放或随机比例缩放。
@register("ReDisoP", "reward_out")
def re_diso_p(reward, ctx, cfg):
    """Period: irregular modification — 50/50 sign-flip-scaled OR random-scale."""
    timestep = current_step()
    if not trigger("ReDisoP", timestep):
        return reward
    # 统一裁到非负；intensity=0 时两个分支都应退化为原 reward。
    intensity = max(float(cfg.intensity), 0.0)
    if ctx.rng.random() < 0.5:
        # intensity=0 -> original reward; intensity=1 -> full sign flip.
        return float(reward) * (1.0 - 2.0 * intensity)
    scale = 1.0 + ((ctx.rng.random() * 2.0) - 1.0) * intensity
    return float(reward) * scale
