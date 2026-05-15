"""Phase 3.3 P2 operator tests.

3 operators: ESRemS, SaSDisoS, SaSRepP. All on es_sample hook.
"""
import numpy as np
import pytest
import torch
from torch.distributions import MultivariateNormal

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


def _mk_dist_and_sample(action_std=0.2, seed=42):
    """Construct a MultivariateNormal mirroring ActorCritic at parameters.py
    defaults: action_dim=2, action_std_init=0.2, batch dim 1.
    """
    torch.manual_seed(seed)
    mean = torch.tensor([[0.3, -0.5]])
    cov_var = torch.full((2,), action_std)
    cov_mat = torch.diag(cov_var).unsqueeze(0)  # (1, 2, 2)
    dist = MultivariateNormal(mean, cov_mat)
    action = dist.sample()
    return action, mean, cov_mat, dist


# ---------------- ESRemS ---------------- #

def test_es_rem_s_replaces_with_mean():
    _set_op("ESRemS")
    action, mean, cov_mat, dist = _mk_dist_and_sample()
    out = mutation.es_sample(action, mean, cov_mat, dist)
    assert torch.equal(out, mean.detach())
    # Sanity: sample is generally not equal to mean
    assert not torch.equal(action, mean)


def test_es_rem_s_preserves_shape_dtype():
    _set_op("ESRemS")
    action, mean, cov_mat, dist = _mk_dist_and_sample()
    out = mutation.es_sample(action, mean, cov_mat, dist)
    assert out.shape == action.shape
    assert out.dtype == action.dtype


# ---------------- SaSDisoS ---------------- #

def test_sas_diso_s_permutes_action_dims():
    _set_op("SaSDisoS", seed=42)
    action, mean, cov_mat, dist = _mk_dist_and_sample()
    # action shape (1, 2); permuting 2 dims may swap or keep
    # Run many times: at least once it should swap (50% chance each)
    swapped = 0
    for _ in range(20):
        out = mutation.es_sample(action, mean, cov_mat, dist)
        # Output should contain same values, just possibly reordered
        out_sorted = torch.sort(out.flatten())[0]
        action_sorted = torch.sort(action.flatten())[0]
        assert torch.equal(out_sorted, action_sorted)
        if not torch.equal(out, action):
            swapped += 1
    # 20 trials, prob swap each ~0.5 -> at least 5 should swap
    assert swapped >= 3


def test_sas_diso_s_preserves_shape_dtype():
    _set_op("SaSDisoS", seed=0)
    action, mean, cov_mat, dist = _mk_dist_and_sample()
    out = mutation.es_sample(action, mean, cov_mat, dist)
    assert out.shape == action.shape
    assert out.dtype == action.dtype


# ---------------- SaSRepP ---------------- #

def test_sas_rep_p_replays_step9_at_step10():
    _set_op("SaSRepP")
    action_step9 = None
    for i in range(1, 10):
        mutation.tick()
        action, mean, cov_mat, dist = _mk_dist_and_sample(seed=i)
        out = mutation.es_sample(action, mean, cov_mat, dist)
        assert torch.equal(out, action)
        if i == 9:
            action_step9 = action.detach().clone()
    # Step 10: trigger
    mutation.tick()
    action_step10, mean, cov_mat, dist = _mk_dist_and_sample(seed=10)
    out = mutation.es_sample(action_step10, mean, cov_mat, dist)
    assert torch.equal(out, action_step9)
    assert not torch.equal(out, action_step10)


def test_sas_rep_p_preserves_shape_dtype():
    _set_op("SaSRepP")
    for i in range(1, 25):
        mutation.tick()
        action, mean, cov_mat, dist = _mk_dist_and_sample(seed=i)
        out = mutation.es_sample(action, mean, cov_mat, dist)
        assert out.shape == action.shape
        assert out.dtype == action.dtype


# ---------------- log_prob recompute consistency ---------------- #

def test_es_operators_allow_caller_to_recompute_log_prob():
    """Audit R1: caller (ppo.py) is responsible for re-evaluating
    dist.log_prob(returned_action) AFTER this hook returns. Operators here
    only need to return a valid tensor of the same shape; we verify shape
    here, the actual recompute happens in ppo.py:73."""
    for op in ("ESRemS", "SaSDisoS", "SaSRepP"):
        reset_ctx_all()
        _set_op(op, seed=0)
        mutation.tick()
        action, mean, cov_mat, dist = _mk_dist_and_sample(seed=0)
        out = mutation.es_sample(action, mean, cov_mat, dist)
        # caller does: log_prob = dist.log_prob(out)
        lp = dist.log_prob(out)
        assert lp.shape == torch.Size([1])
        assert torch.isfinite(lp).all()


# ---------------- Meta ---------------- #

def test_all_p2_operators_registered():
    assert "ESRemS" in mutation.list_operators("es_sample")
    assert "SaSDisoS" in mutation.list_operators("es_sample")
    assert "SaSRepP" in mutation.list_operators("es_sample")
