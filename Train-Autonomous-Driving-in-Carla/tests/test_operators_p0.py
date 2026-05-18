"""Phase 3.1 P0 operator tests.

6 operators x ~3 cases each = ~18 tests.
Coverage per operator: active semantic + dtype/shape preservation.
The MUTATION_TYPE=none identity path is already covered by
test_mutation_hooks.py::test_<hook>_none_identity.
"""
import numpy as np
import pytest

import mutation
from mutation.config import MutationConfig, reset_config
from mutation.context import reset_all as reset_ctx_all


def _set_op(op, seed=0, intensity=1.0):
    reset_config(MutationConfig(mutation_type=op, seed=seed, intensity=intensity))


@pytest.fixture(autouse=True)
def reset_mut():
    reset_config(MutationConfig(mutation_type="none", seed=0, intensity=0.0))
    reset_ctx_all()
    yield
    reset_config(None)
    reset_ctx_all()


def _mk_obs(rng_seed=0):
    rs_img = np.random.RandomState(rng_seed)
    rs_nav = np.random.RandomState(rng_seed + 1)
    img = rs_img.rand(160, 80, 3).astype(np.float32)
    nav = rs_nav.rand(5)
    return [img, nav]


# ---------------- StRepP ---------------- #

def test_st_rep_p_replays_step9_at_step10():
    _set_op("StRepP")
    # Steps 1..9: non-trigger; caches obs
    obs9 = None
    for i in range(1, 10):
        mutation.tick()
        obs = _mk_obs(rng_seed=i)
        out = mutation.state_out(obs)
        assert np.array_equal(out[0], obs[0]) and np.array_equal(out[1], obs[1])
        if i == 9:
            obs9 = obs

    # Step 10: trigger; should return step 9 cached
    mutation.tick()
    obs10 = _mk_obs(rng_seed=10)
    out = mutation.state_out(obs10)
    assert np.array_equal(out[1], obs9[1])
    assert not np.array_equal(out[1], obs10[1])


def test_st_rep_p_shape_dtype_preserved():
    _set_op("StRepP")
    for i in range(1, 25):
        mutation.tick()
        obs = _mk_obs(rng_seed=i)
        out = mutation.state_out(obs)
        assert out[0].dtype == obs[0].dtype
        assert out[0].shape == obs[0].shape
        assert out[1].dtype == obs[1].dtype
        assert out[1].shape == obs[1].shape


# ---------------- StFuzS ---------------- #

def test_st_fuz_s_quantizes_nav():
    _set_op("StFuzS", intensity=1.0)
    obs = _mk_obs(rng_seed=5)
    mutation.tick()
    out = mutation.state_out(obs)
    assert not np.array_equal(out[1], obs[1])


def test_st_fuz_s_quantizes_image_in_normalized_domain():
    """phase3.6 fix: quantize in [0,1] float domain, NOT via int32 cast.

    Guard against the regression: nonzero [0,1] inputs must stay mostly nonzero
    (the old int32 cast bug produced an all-black image = visual removal, not
    fuzzing).
    """
    _set_op("StFuzS", intensity=1.0)
    obs = _mk_obs(rng_seed=5)
    mutation.tick()
    out = mutation.state_out(obs)
    # critical anti-regression: nonzero input must produce mostly-nonzero output
    nonzero_ratio = (out[0] > 0).sum() / out[0].size
    assert nonzero_ratio > 0.9, \
        f"StFuzS turned image black (cast int32 bug regressed?): nonzero ratio = {nonzero_ratio}"
    # output stays in [0, 1] (still normalized)
    assert out[0].min() >= 0.0 and out[0].max() <= 1.0
    # intensity=1.0 -> img_bits=4 -> img_levels=16, so unique values should be at most 17
    # (the values are {0/16, 1/16, ..., 16/16})
    unique_vals = np.unique(out[0])
    assert len(unique_vals) <= 17, \
        f"StFuzS quantization too fine: {len(unique_vals)} unique values, expected <= 17"


def test_st_fuz_s_shape_dtype_preserved():
    _set_op("StFuzS", intensity=0.5)
    obs = _mk_obs(rng_seed=7)
    mutation.tick()
    out = mutation.state_out(obs)
    assert out[0].dtype == obs[0].dtype
    assert out[0].shape == obs[0].shape
    assert out[1].dtype == obs[1].dtype
    assert out[1].shape == obs[1].shape


# ---------------- ReRepP ---------------- #

def test_re_rep_p_replays_step9_at_step10():
    _set_op("ReRepP")
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for r in rewards:
        mutation.tick()
        assert mutation.reward_out(r) == r
    mutation.tick()
    assert mutation.reward_out(1.0) == 0.9


def test_re_rep_p_type_preserved():
    _set_op("ReRepP")
    mutation.tick()
    out = mutation.reward_out(0.5)
    assert isinstance(out, (int, float))


# ---------------- ReDistP ---------------- #

def test_re_dist_p_passes_through_at_non_trigger():
    _set_op("ReDistP", intensity=1.0)
    for _ in range(9):
        mutation.tick()
        assert mutation.reward_out(0.5) == 0.5


def test_re_dist_p_adds_noise_at_trigger():
    _set_op("ReDistP", seed=42, intensity=1.0)
    for _ in range(10):
        mutation.tick()
    out = mutation.reward_out(0.0)
    assert isinstance(out, float)
    assert abs(out) > 1e-8  # noise std=0.5 -> very unlikely to hit exactly 0


def test_re_dist_p_seeded_reproducible():
    _set_op("ReDistP", seed=42, intensity=1.0)
    for _ in range(10):
        mutation.tick()
    out1 = mutation.reward_out(0.0)

    reset_ctx_all()
    _set_op("ReDistP", seed=42, intensity=1.0)
    for _ in range(10):
        mutation.tick()
    out2 = mutation.reward_out(0.0)
    assert out1 == out2


# ---------------- AcRepR ---------------- #

def test_ac_rep_r_first_call_caches_and_passes():
    _set_op("AcRepR", seed=42, intensity=1.0)
    a0 = np.array([0.1, 0.2])
    out0 = mutation.action_in(a0)
    assert np.array_equal(out0, a0)


def test_ac_rep_r_trigger_rate_in_range():
    _set_op("AcRepR", seed=42, intensity=1.0)
    # Seed first cache
    mutation.action_in(np.array([0.0, 0.0]))
    triggered = 0
    for i in range(200):
        a = np.array([float(i) / 200, -float(i) / 200])
        out = mutation.action_in(a)
        if not np.array_equal(out, a):
            triggered += 1
    # default prob=0.3 -> expect ~60/200; allow [40, 90] for noise tolerance
    assert 40 <= triggered <= 90


def test_ac_rep_r_shape_dtype_preserved():
    _set_op("AcRepR", seed=0)
    a = np.array([0.5, -0.5])
    for _ in range(30):
        out = mutation.action_in(a)
        assert out.dtype == a.dtype
        assert out.shape == a.shape


# ---------------- AcFuzS ---------------- #

def test_ac_fuz_s_quantizes_to_expected_bins():
    _set_op("AcFuzS", intensity=1.0)
    # intensity=1 -> bins=max(int(10*(1-0.5)), 2)=5
    a = np.array([0.123, -0.456])
    out = mutation.action_in(a)
    expected = np.round(a * 5) / 5
    assert np.allclose(out, expected)


def test_ac_fuz_s_shape_dtype_preserved():
    _set_op("AcFuzS", intensity=0.5)
    a = np.array([0.3, -0.7])
    out = mutation.action_in(a)
    assert out.dtype == a.dtype
    assert out.shape == a.shape


# ---------------- Meta: P0 operators all registered ---------------- #

def test_all_p0_operators_registered():
    assert set(mutation.list_operators("state_out")) >= {"StRepP", "StFuzS"}
    assert set(mutation.list_operators("reward_out")) >= {"ReRepP", "ReDistP"}
    assert set(mutation.list_operators("action_in")) >= {"AcRepR", "AcFuzS"}
