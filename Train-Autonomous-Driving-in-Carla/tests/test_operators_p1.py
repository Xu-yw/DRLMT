"""Phase 3.2 P1 operator tests.

5 operators: StDistP, StDisoP, ReDisoP, AcDisoR, PVDistR.
Each gets ~2-3 cases: active semantic + dtype/shape preservation + (where
relevant) seed reproducibility.
"""
import numpy as np
import pytest
import torch

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


# ---------------- StDistP ---------------- #

def test_st_dist_p_adds_noise_at_trigger():
    _set_op("StDistP", seed=42, intensity=1.0)
    for _ in range(9):
        mutation.tick()
    # Non-trigger: identity
    obs9 = _mk_obs(9)
    out9 = mutation.state_out(obs9)
    assert np.allclose(out9[1], obs9[1])
    # Trigger
    mutation.tick()
    obs10 = _mk_obs(10)
    out10 = mutation.state_out(obs10)
    assert not np.allclose(out10[1], obs10[1])


def test_st_dist_p_shape_dtype_preserved():
    _set_op("StDistP", seed=0, intensity=1.0)
    for _ in range(15):
        mutation.tick()
        obs = _mk_obs()
        out = mutation.state_out(obs)
        assert out[0].dtype == obs[0].dtype
        assert out[0].shape == obs[0].shape
        assert out[1].dtype == obs[1].dtype
        assert out[1].shape == obs[1].shape


# ---------------- StDisoP ---------------- #

def test_st_diso_p_replaces_with_history():
    _set_op("StDisoP", seed=42, intensity=1.0)
    # Fill history with 9 distinct obs (steps 1..9, non-trigger)
    cached_navs = []
    for i in range(1, 10):
        mutation.tick()
        obs = _mk_obs(rng_seed=i)
        out = mutation.state_out(obs)
        cached_navs.append(obs[1].copy())
        assert np.array_equal(out[1], obs[1])  # passthrough on non-trigger
    # Trigger at step 10: should pick from history (cached_navs[0..8])
    mutation.tick()
    obs10 = _mk_obs(rng_seed=10)
    out10 = mutation.state_out(obs10)
    # Output nav should match one of the cached obs (not obs10's own)
    matched_history = any(np.array_equal(out10[1], c) for c in cached_navs)
    assert matched_history
    assert not np.array_equal(out10[1], obs10[1])


def test_st_diso_p_shape_dtype_preserved():
    _set_op("StDisoP", seed=0, intensity=1.0)
    for i in range(1, 25):
        mutation.tick()
        obs = _mk_obs(rng_seed=i)
        out = mutation.state_out(obs)
        assert out[0].dtype == obs[0].dtype
        assert out[0].shape == obs[0].shape
        assert out[1].dtype == obs[1].dtype


# ---------------- ReDisoP ---------------- #

def test_re_diso_p_modifies_at_trigger():
    _set_op("ReDisoP", seed=42, intensity=1.0)
    for _ in range(9):
        mutation.tick()
    # Non-trigger: identity
    assert mutation.reward_out(2.0) == 2.0
    # Trigger
    mutation.tick()
    out = mutation.reward_out(2.0)
    # Either sign-flipped (-2.0) or scaled (2.0 * random*2 in [0, 4])
    assert out != 2.0 or out == 0.0  # small chance of scale=1 too


def test_re_diso_p_seeded_reproducible():
    _set_op("ReDisoP", seed=42, intensity=1.0)
    for _ in range(10):
        mutation.tick()
    out1 = mutation.reward_out(3.0)
    reset_ctx_all()
    _set_op("ReDisoP", seed=42, intensity=1.0)
    for _ in range(10):
        mutation.tick()
    out2 = mutation.reward_out(3.0)
    assert out1 == out2


# ---------------- AcDisoR ---------------- #

def test_ac_diso_r_replaces_with_random_at_trigger():
    _set_op("AcDisoR", seed=42, intensity=1.0)
    triggered = 0
    inputs = []
    outputs = []
    for i in range(200):
        a = np.array([float(i) / 200, -float(i) / 200])
        out = mutation.action_in(a)
        inputs.append(a)
        outputs.append(out)
        if not np.array_equal(out, a):
            triggered += 1
            # Replacement should be in [-1, 1]
            assert -1.0 <= out[0] <= 1.0 and -1.0 <= out[1] <= 1.0
    # prob=0.3 -> expect ~60/200; allow [40, 90]
    assert 40 <= triggered <= 90


def test_ac_diso_r_shape_dtype_preserved():
    _set_op("AcDisoR", seed=0, intensity=1.0)
    a = np.array([0.5, -0.5])
    for _ in range(30):
        out = mutation.action_in(a)
        assert out.dtype == a.dtype
        assert out.shape == a.shape


# ---------------- PVDistR ---------------- #

def _mk_pv_dist(mean_value=0.0, std=0.2):
    """Build a MultivariateNormal matching PPO's actor distribution shape."""
    from torch.distributions import MultivariateNormal
    mean = torch.tensor([[mean_value, mean_value]], dtype=torch.float32)
    cov_mat = torch.diag(torch.tensor([std * std, std * std], dtype=torch.float32))
    return MultivariateNormal(mean, cov_mat)


def test_pv_dist_r_passes_through_when_not_triggered():
    _set_op("PVDistR", seed=42, intensity=0.0)
    # intensity=0 -> prob=0.3 still positive but noise_std=0
    action = torch.tensor([[0.3, -0.5]])
    logprob = torch.tensor([1.234])
    a2, lp2 = mutation.pv_out(action, logprob, None)
    # With std=0, even if triggered, noise is 0 -> action unchanged
    assert torch.allclose(a2, action)
    assert torch.equal(lp2, logprob)


def test_pv_dist_r_adds_noise_some_steps_eval_path():
    """Evaluation path (dist=None): noise applied to action, logprob unchanged (placeholder)."""
    _set_op("PVDistR", seed=42, intensity=1.0)
    action = torch.tensor([[0.3, -0.5]])
    logprob = torch.tensor([1.234])
    triggered = 0
    for _ in range(200):
        a2, lp2 = mutation.pv_out(action, logprob, None)
        if not torch.allclose(a2, action):
            triggered += 1
        # Evaluation path: logprob is a placeholder, never recomputed
        assert torch.equal(lp2, logprob)
    # prob=0.3 -> ~60/200; allow [40, 90]
    assert 40 <= triggered <= 90


def test_pv_dist_r_recomputes_logprob_on_training_path():
    """Training path (dist provided): on trigger, logprob = dist.log_prob(new_action)."""
    _set_op("PVDistR", seed=42, intensity=1.0)
    action = torch.tensor([[0.3, -0.5]])
    logprob = torch.tensor([1.234])
    dist = _mk_pv_dist()
    triggered_recomputed = 0
    untriggered_unchanged = 0
    for _ in range(200):
        a2, lp2 = mutation.pv_out(action, logprob, dist)
        if not torch.allclose(a2, action):
            # On trigger, lp2 must equal dist.log_prob(a2), NOT the original logprob
            expected = dist.log_prob(a2).detach()
            assert torch.allclose(lp2, expected), \
                f"logprob not recomputed: got {lp2}, expected {expected}"
            triggered_recomputed += 1
        else:
            # No trigger: logprob unchanged
            assert torch.equal(lp2, logprob)
            untriggered_unchanged += 1
    # prob=0.3 -> ~60/200 triggered; allow [40, 90]
    assert 40 <= triggered_recomputed <= 90
    assert triggered_recomputed + untriggered_unchanged == 200


def test_pv_dist_r_shape_dtype_preserved():
    _set_op("PVDistR", seed=0, intensity=1.0)
    action = torch.tensor([[0.3, -0.5]], dtype=torch.float32)
    logprob = torch.tensor([0.5], dtype=torch.float32)
    a2, lp2 = mutation.pv_out(action, logprob, None)
    assert a2.shape == action.shape
    assert a2.dtype == action.dtype
    assert lp2.dtype == logprob.dtype


# ---------------- Meta ---------------- #

def test_all_p1_operators_registered():
    assert "StDistP" in mutation.list_operators("state_out")
    assert "StDisoP" in mutation.list_operators("state_out")
    assert "ReDisoP" in mutation.list_operators("reward_out")
    assert "AcDisoR" in mutation.list_operators("action_in")
    assert "PVDistR" in mutation.list_operators("pv_out")
