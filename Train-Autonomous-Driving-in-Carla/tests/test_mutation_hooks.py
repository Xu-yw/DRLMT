"""Phase 2 + Phase 3.0a hook tests.

Coverage:
  - Phase 2: 5 hooks x {none, sentinel active no-op} = 10 identity checks
             + config env loading (default / override)
  - Phase 3.0a: pv_out hook, timing (P/R/S), context lifecycle, registry dispatch
"""
import numpy as np
import pytest

import mutation
from mutation.config import MutationConfig, reset_config
from mutation.context import reset_all as reset_ctx_all
from mutation.registry import reset_registry


@pytest.fixture(autouse=True)
def reset_mutation_state():
    reset_config(MutationConfig(mutation_type="none", seed=0, intensity=0.0))
    reset_ctx_all()
    reset_registry()
    yield
    reset_config(None)
    reset_ctx_all()
    reset_registry()


def _set_active(op="test_sentinel", seed=42, intensity=0.5):
    reset_config(MutationConfig(mutation_type=op, seed=seed, intensity=intensity))


# ---------------- Phase 2 hook identity ---------------- #

def test_state_out_none_identity():
    obs = [np.zeros((160, 80, 3), dtype=np.float32),
           np.array([0.1, 0.2, 0.3, 0.4, 0.5])]
    assert mutation.state_out(obs) is obs


def test_state_out_active_no_operator_passthrough():
    _set_active()
    obs = [np.ones((160, 80, 3), dtype=np.float32),
           np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
    assert mutation.state_out(obs) is obs


def test_action_in_none_identity():
    action = np.array([0.3, 0.7])
    assert mutation.action_in(action) is action


def test_action_in_active_no_operator_passthrough():
    _set_active()
    action = np.array([0.3, 0.7])
    assert mutation.action_in(action) is action


def test_reward_out_none_identity():
    assert mutation.reward_out(0.5) == 0.5
    assert mutation.reward_out(-10) == -10


def test_reward_out_active_no_operator_passthrough():
    _set_active()
    assert mutation.reward_out(0.5) == 0.5


def test_rc_none_returns_default():
    assert mutation.rc("cruise", 1.0) == 1.0
    assert mutation.rc("progress", 0.05) == 0.05
    assert mutation.rc("offcent", 0.2) == 0.2


def test_rc_active_no_operator_returns_default():
    _set_active()
    assert mutation.rc("cruise", 1.0) == 1.0


def test_es_sample_none_identity():
    sentinel = object()
    assert mutation.es_sample(sentinel, mean=None, cov_mat=None, dist=None) is sentinel


def test_es_sample_active_no_operator_passthrough():
    _set_active()
    sentinel = object()
    assert mutation.es_sample(sentinel, mean=None, cov_mat=None, dist=None) is sentinel


def test_config_from_env_default(monkeypatch):
    for k in ["MUTATION_TYPE", "MUTATION_SEED", "MUTATION_INTENSITY"]:
        monkeypatch.delenv(k, raising=False)
    reset_config(None)
    cfg = mutation.get_config()
    assert cfg.mutation_type == "none"
    assert cfg.seed == 0
    assert cfg.intensity == 1.0
    assert cfg.is_active is False


def test_config_from_env_active(monkeypatch):
    monkeypatch.setenv("MUTATION_TYPE", "StRepP")
    monkeypatch.setenv("MUTATION_SEED", "7")
    monkeypatch.setenv("MUTATION_INTENSITY", "0.3")
    reset_config(None)
    cfg = mutation.get_config()
    assert cfg.mutation_type == "StRepP"
    assert cfg.seed == 7
    assert cfg.intensity == 0.3
    assert cfg.is_active is True


# ---------------- Phase 3.0a: pv_out hook (phase3.6 added dist param) ---------------- #

def test_pv_out_none_identity():
    a, lp = object(), object()
    a2, lp2 = mutation.pv_out(a, lp, None)
    assert a2 is a
    assert lp2 is lp


def test_pv_out_active_no_operator_passthrough():
    _set_active()
    a, lp = object(), object()
    a2, lp2 = mutation.pv_out(a, lp, None)
    assert a2 is a
    assert lp2 is lp


def test_pv_out_dispatcher_passes_dist_to_operator():
    """Verify dispatcher forwards dist (training path arg) to registered operator."""
    captured = {}

    @mutation.register("PVCapturingOp", "pv_out")
    def _cap(action, logprob, dist, ctx, cfg):
        captured["dist"] = dist
        return action, logprob

    _set_active(op="PVCapturingOp")
    sentinel_dist = object()
    mutation.pv_out("a", "lp", sentinel_dist)
    assert captured["dist"] is sentinel_dist
    # Eval path: None is also passed through
    mutation.pv_out("a", "lp", None)
    assert captured["dist"] is None


# ---------------- Phase 3.0a: timing ---------------- #

def test_trigger_periodical():
    assert mutation.trigger("StRepP", 0) is False
    assert mutation.trigger("StRepP", 9) is False
    assert mutation.trigger("StRepP", 10) is True
    assert mutation.trigger("StRepP", 20) is True
    assert mutation.trigger("StRepP", 25) is False


def test_trigger_periodical_custom_period():
    assert mutation.trigger("StRepP", 4, period_steps=5) is False
    assert mutation.trigger("StRepP", 5, period_steps=5) is True


def test_trigger_random_with_rng():
    import random
    rng = random.Random(0)
    fires = sum(mutation.trigger("AcDisoR", t, rng=rng, prob=0.3) for t in range(1000))
    assert 250 <= fires <= 350  # binomial ~300


def test_trigger_random_without_rng_raises():
    with pytest.raises(ValueError):
        mutation.trigger("AcDisoR", 5)


def test_trigger_sustained():
    for t in range(20):
        assert mutation.trigger("StFuzS", t) is True


def test_trigger_unknown_suffix_raises():
    with pytest.raises(ValueError):
        mutation.trigger("BogusX", 1)


# ---------------- Phase 3.0a: context ---------------- #

def test_context_singleton_per_op():
    a = mutation.get_context("StRepP", seed=1)
    b = mutation.get_context("StRepP", seed=999)  # seed ignored after first call
    assert a is b


def test_context_isolated_rngs():
    a = mutation.get_context("StRepP", seed=1)
    b = mutation.get_context("ReDistP", seed=1)
    # same seed -> same rng state -> same sequence (proves isolation, not collision)
    seq_a = [a.rng.random() for _ in range(5)]
    seq_b = [b.rng.random() for _ in range(5)]
    assert seq_a == seq_b
    # advancing a should not advance b
    a.rng.random()
    assert a.rng.random() != b.rng.random()


def test_context_state_dict_isolated():
    a = mutation.get_context("StRepP", seed=0)
    b = mutation.get_context("ReRepP", seed=0)
    a.state["last"] = 42
    assert "last" not in b.state


def test_tick_advances_global_step():
    assert mutation.current_step() == 0
    mutation.tick()
    mutation.tick()
    assert mutation.current_step() == 2


def test_reset_episode_clears_state_keeps_rng():
    ctx = mutation.get_context("StRepP", seed=0)
    ctx.state["last"] = "cached"
    rng_state_before = ctx.rng.getstate()
    mutation.tick(); mutation.tick(); mutation.tick()
    assert mutation.current_step() == 3
    mutation.reset_episode()
    assert ctx.state == {}
    assert ctx.rng.getstate() == rng_state_before
    assert mutation.current_step() == 0


def test_action_in_ticks_step():
    assert mutation.current_step() == 0
    mutation.action_in(np.array([0.0, 0.0]))
    mutation.action_in(np.array([0.0, 0.0]))
    assert mutation.current_step() == 2


# ---------------- Phase 3.0a: registry dispatch ---------------- #

def test_registry_register_and_get():
    @mutation.register("DummyOp", "state_out")
    def _dummy(obs, ctx, cfg):
        return "mutated_obs"

    fn = mutation.get_operator("state_out", "DummyOp")
    assert fn is _dummy
    # Use containment: baseline already has bundled Phase 3.1+ state_out ops.
    assert "DummyOp" in mutation.list_operators("state_out")


def test_dispatch_routes_to_registered_operator():
    @mutation.register("DoubleReward", "reward_out")
    def _double(reward, ctx, cfg):
        return reward * 2

    _set_active(op="DoubleReward")
    assert mutation.reward_out(3.0) == 6.0


def test_dispatch_falls_back_when_op_not_in_category():
    @mutation.register("OnlyStateOp", "state_out")
    def _x(obs, ctx, cfg):
        return "mutated"

    _set_active(op="OnlyStateOp")
    # OnlyStateOp registered for state_out but NOT reward_out -> passthrough
    assert mutation.reward_out(5.0) == 5.0
    # state_out should dispatch
    assert mutation.state_out("x") == "mutated"


def test_dispatch_passes_ctx_and_cfg():
    captured = {}

    @mutation.register("CapturingOp", "action_in")
    def _cap(action, ctx, cfg):
        captured["ctx"] = ctx
        captured["cfg"] = cfg
        return action

    _set_active(op="CapturingOp", seed=123, intensity=0.5)
    arr = np.array([0.1, 0.2])
    mutation.action_in(arr)
    assert captured["ctx"].op_name == "CapturingOp"
    assert captured["ctx"].seed == 123
    assert captured["cfg"].intensity == 0.5
