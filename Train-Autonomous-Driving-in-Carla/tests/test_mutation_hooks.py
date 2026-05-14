"""Phase 2 hook on/off path tests.

Coverage:
  - 5 hooks x {none, active sentinel} = 10 identity checks
  - Singleton config reset between tests
  - Env var loading both default and override
"""
import numpy as np
import pytest

import mutation
from mutation.config import MutationConfig, reset_config


@pytest.fixture(autouse=True)
def reset_mutation_config():
    reset_config(MutationConfig(mutation_type="none", seed=0, intensity=0.0))
    yield
    reset_config(None)


def _set_active():
    reset_config(MutationConfig(mutation_type="test_sentinel", seed=42, intensity=0.5))


def test_state_out_none_identity():
    obs = [
        np.zeros((160, 80, 3), dtype=np.float32),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    ]
    assert mutation.state_out(obs) is obs


def test_state_out_active_passthrough():
    _set_active()
    obs = [
        np.ones((160, 80, 3), dtype=np.float32),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    ]
    assert mutation.state_out(obs) is obs


def test_action_in_none_identity():
    action = np.array([0.3, 0.7])
    assert mutation.action_in(action) is action


def test_action_in_active_passthrough():
    _set_active()
    action = np.array([0.3, 0.7])
    assert mutation.action_in(action) is action


def test_reward_out_none_identity():
    assert mutation.reward_out(0.5) == 0.5
    assert mutation.reward_out(-10) == -10


def test_reward_out_active_passthrough():
    _set_active()
    assert mutation.reward_out(0.5) == 0.5


def test_rc_none_returns_default():
    assert mutation.rc("cruise", 1.0) == 1.0
    assert mutation.rc("progress", 0.05) == 0.05
    assert mutation.rc("offcent", 0.2) == 0.2


def test_rc_active_passthrough():
    _set_active()
    assert mutation.rc("cruise", 1.0) == 1.0
    assert mutation.rc("progress", 0.05) == 0.05


def test_es_sample_none_identity():
    sentinel = object()
    out = mutation.es_sample(sentinel, mean=None, cov_mat=None, dist=None)
    assert out is sentinel


def test_es_sample_active_passthrough():
    _set_active()
    sentinel = object()
    out = mutation.es_sample(sentinel, mean=None, cov_mat=None, dist=None)
    assert out is sentinel


def test_config_from_env_default(monkeypatch):
    for k in ["MUTATION_TYPE", "MUTATION_SEED", "MUTATION_INTENSITY"]:
        monkeypatch.delenv(k, raising=False)
    reset_config(None)
    cfg = mutation.get_config()
    assert cfg.mutation_type == "none"
    assert cfg.seed == 0
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
