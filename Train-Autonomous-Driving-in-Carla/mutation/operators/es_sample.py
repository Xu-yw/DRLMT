"""Exploration-sample operators (hook category = 'es_sample').

P2 (Phase 3.3) -- inject at ppo.py:72-73 (between dist.sample and
dist.log_prob). Caller (ppo.py) reassigns log_prob = dist.log_prob(returned)
AFTER this hook returns, so (action, log_prob) self-consistency is automatic
regardless of what the operator does (audit R1).

Hook signature: fn(action, mean, cov_mat, dist, ctx, cfg) -> action
  action: dist.sample() output, shape (1, 2) tensor
  mean: ActorCritic.actor(obs), shape (1, 2)
  cov_mat: shape (1, 2, 2) registered buffer
  dist: MultivariateNormal(mean, cov_mat) instance

Operators:
  ESRemS  -- Permanently missing exploration: replace sample with dist.mean.

NOTE (2026-05-16): SaSDisoS / SaSRepP were removed.
Reason: per DRLMutation paper page 8/16, SaS = "sampling strategy" of the
replay buffer (off-policy concept). PPO is on-policy with no replay buffer,
so SaS has no direct correspondence in this codebase. The earlier
implementation engineered SaS into action-sampling permutation/replay, which
is a name-only match and breaks faithful paper alignment. We remove the two
operators rather than retain a misleading mapping. See plan-06 and the
final report (Phase 8) for the methodological rationale.
"""

# 中文说明：本文件实现 Exploration Strategy 相关 mutant。
# 在 PPO 中，论文里的 ES 可以映射到动作分布采样策略：actor 输出 mean，代码构造 MultivariateNormal，
# 然后从 dist.sample() 得到随机动作。es_sample hook 就插在 sample 之后、log_prob 计算之前。
# 因为 caller 会在 hook 返回后重新 dist.log_prob(action)，所以此处替换 action 仍能保持 logprob 自洽。

from mutation.registry import register


# 中文注释：ESRemS，Exploration Strategy Remove Sustained；持续移除随机探索采样。
@register("ESRemS", "es_sample")
def es_rem_s(action, mean, cov_mat, dist, ctx, cfg):
    """Sustained: always return mean (no exploration)."""
    # 用 actor mean 替代 dist.sample()，等价于不再进行 stochastic exploration。
    return mean.detach()
