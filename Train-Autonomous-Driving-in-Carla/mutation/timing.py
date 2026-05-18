"""Timing dispatcher: P (periodical), R (random), S (sustained).

Operator name suffix encodes the timing:
  *P -> periodical: trigger if timestep > 0 and timestep % period_steps == 0
  *R -> random:     trigger if rng.random() < prob
  *S -> sustained:  always trigger

Default knobs (03-operators-spec.md):
  period_steps = 10
  prob         = 0.3
"""

# 中文说明：本文件把算子名最后一个字母解释成触发策略。
# P = Periodical，周期触发；R = Random，按概率随机触发；S = Sustained，每一步都触发。
# current_step 由 action_in() 的 tick() 推进，所以周期类算子的“第 10 步、第 20 步”对应 env.step 入口次数。
# 注意当前默认周期固定为 10；若某算子未来需要 intensity 调周期，需要显式传 period_steps。

DEFAULT_PERIOD_STEPS = 10
DEFAULT_PROB = 0.3


# 中文注释：所有算子共用的触发判断；op_name 后缀决定解释哪一种 timing。
def trigger(op_name, timestep, rng=None, period_steps=DEFAULT_PERIOD_STEPS, prob=DEFAULT_PROB):
    suffix = op_name[-1]
    if suffix == "P":
        return timestep > 0 and (timestep % period_steps == 0)
    if suffix == "R":
        if rng is None:
            raise ValueError("R-suffix operator requires rng")
        return rng.random() < prob
    if suffix == "S":
        return True
    raise ValueError(f"unknown timing suffix in {op_name!r}: {suffix!r}")
