"""Phase 3.3 P2 operator tests.

1 operator: ESRemS (es_sample hook).

NOTE (2026-05-16): SaSDisoS / SaSRepP tests removed alongside operator
deletion. See mutation/operators/es_sample.py module docstring for the
PPO on-policy / paper off-policy rationale.
"""
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


# ---------------- log_prob recompute consistency ---------------- #

def test_es_operators_allow_caller_to_recompute_log_prob():
    """Audit R1: caller (ppo.py) is responsible for re-evaluating
    dist.log_prob(returned_action) AFTER this hook returns. Operators here
    only need to return a valid tensor of the same shape; we verify shape
    here, the actual recompute happens in ppo.py:73."""
    reset_ctx_all()
    _set_op("ESRemS", seed=0)
    mutation.tick()
    action, mean, cov_mat, dist = _mk_dist_and_sample(seed=0)
    out = mutation.es_sample(action, mean, cov_mat, dist)
    # caller does: log_prob = dist.log_prob(out)
    lp = dist.log_prob(out)
    assert lp.shape == torch.Size([1])
    assert torch.isfinite(lp).all()


# ---------------- Meta ---------------- #

def test_es_operators_registered_no_sas():
    """ESRemS present; SaSDisoS / SaSRepP must NOT be registered."""
    es_ops = mutation.list_operators("es_sample")
    assert "ESRemS" in es_ops
    assert "SaSDisoS" not in es_ops
    assert "SaSRepP" not in es_ops
