"""P0-1: mutation runtime probe — first-call evidence emission tests.

Covers:
  - All 5 data hooks emit [MUT-PROBE] line on first active dispatch
  - rc hook emits with coef_name as part of the dedup key
  - perturbed=False detection (regression of the 2026-05-18 intensity=0 bug)
  - sanity_nonzero detection (regression of the 2026-05-18 StFuzS int32 blinding bug)
  - MUTATION_TYPE=none and unregistered-op-category paths emit no probe
  - One-shot semantics: subsequent calls of the same (hook, op) do NOT re-emit
  - reset() clears emission state (test helper)
"""
import re

import numpy as np
import pytest

import mutation
from mutation import _probe
from mutation.config import MutationConfig, reset_config
from mutation.context import reset_all as reset_ctx_all
from mutation.registry import reset_registry


@pytest.fixture(autouse=True)
def _isolate():
    reset_config(MutationConfig(mutation_type="none", seed=0, intensity=0.0))
    reset_ctx_all()
    reset_registry()
    _probe.reset()
    yield
    reset_config(None)
    reset_ctx_all()
    reset_registry()
    _probe.reset()


def _set_op(op, seed=0, intensity=1.0):
    reset_config(MutationConfig(mutation_type=op, seed=seed, intensity=intensity))


def _probe_lines(out):
    return [ln for ln in out.splitlines() if "[MUT-PROBE]" in ln]


# ----------- emission happens on first active dispatch ----------- #


def test_state_out_emits_probe_line_with_required_fields(capsys):
    _set_op("StFuzS", intensity=1.0)
    obs = [np.ones((160, 80, 3), dtype=np.float32) * 0.7,
           np.array([0.1, 0.2, 0.3, 0.4, 0.5])]
    mutation.tick()
    mutation.state_out(obs)
    lines = _probe_lines(capsys.readouterr().out)
    assert len(lines) == 1
    ln = lines[0]
    assert "hook=state_out" in ln
    assert "op=StFuzS" in ln
    assert "intensity=1.000" in ln
    assert "perturbed=True" in ln
    assert "img_delta=" in ln and "nav_delta=" in ln
    assert "sanity_nonzero=" in ln
    # StFuzS quantizes to 16 levels on [0,1]; output should still be mostly nonzero
    m = re.search(r"sanity_nonzero=([\d.]+)", ln)
    assert m and float(m.group(1)) > 0.9, f"StFuzS blinding regression suspect: {ln}"


def test_action_in_emits_probe_line(capsys):
    _set_op("AcFuzS", intensity=1.0)
    mutation.action_in(np.array([0.123, -0.456]))
    lines = _probe_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert "hook=action_in" in lines[0]
    assert "op=AcFuzS" in lines[0]
    assert "perturbed=True" in lines[0]


def test_reward_out_emits_probe_line(capsys):
    _set_op("ReRepP", intensity=1.0)
    # ReRepP triggers at step 10 (P suffix = periodic, every 10 ticks)
    for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mutation.tick()
        mutation.reward_out(r)
    mutation.tick()  # step 10
    mutation.reward_out(1.0)
    lines = _probe_lines(capsys.readouterr().out)
    assert len(lines) == 1  # only first active dispatch emits
    assert "hook=reward_out" in lines[0]
    assert "op=ReRepP" in lines[0]


def test_pv_out_emits_probe_line(capsys):
    @mutation.register("PVMock", "pv_out")
    def _mock(action, logprob, dist, ctx, cfg):
        return action + 0.1, logprob + 0.01

    _set_op("PVMock", intensity=1.0)
    action = np.array([0.1, 0.2])
    logprob = 0.0
    mutation.pv_out(action, logprob, dist=None)
    lines = _probe_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert "hook=pv_out" in lines[0]
    assert "op=PVMock" in lines[0]
    assert "perturbed=True" in lines[0]
    assert "logprob_delta=" in lines[0]


def test_es_sample_emits_probe_line(capsys):
    @mutation.register("ESMock", "es_sample")
    def _mock(sample, mean, cov_mat, dist, ctx, cfg):
        return sample + 0.1

    _set_op("ESMock", intensity=1.0)
    mutation.es_sample(np.array([0.0, 0.0]), mean=None, cov_mat=None, dist=None)
    lines = _probe_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert "hook=es_sample" in lines[0]
    assert "op=ESMock" in lines[0]
    assert "perturbed=True" in lines[0]


def test_rc_emits_probe_line_with_coef_name(capsys):
    @mutation.register("RCMock", "rc")
    def _mock(coef_name, default, ctx, cfg):
        return default * 2.0

    _set_op("RCMock", intensity=1.0)
    mutation.rc("cruise", 1.0)
    lines = _probe_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert "hook=rc" in lines[0]
    assert "op=RCMock" in lines[0]
    assert "coef=cruise" in lines[0]
    assert "perturbed=True" in lines[0]


# ----------- regression catchers for the two real bugs ----------- #


def test_state_out_perturbed_false_when_intensity_zero(capsys):
    """Regression: 2026-05-18 event #2. intensity=0 -> StDistP noise std=0 -> no-op.

    The probe must report perturbed=False so wrapper / human can fail-fast.
    """
    _set_op("StDistP", intensity=0.0)
    obs = [np.ones((160, 80, 3), dtype=np.float32) * 0.5,
           np.array([0.1, 0.2, 0.3, 0.4, 0.5])]
    # StDistP is P (periodic, period 10); advance to step 10 to trigger
    for _ in range(10):
        mutation.tick()
    mutation.state_out(obs)
    lines = _probe_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert "perturbed=False" in lines[0], (
        f"intensity=0 must produce perturbed=False (fail-fast signal). got: {lines[0]}"
    )


def test_state_out_sanity_nonzero_catches_blinding(capsys):
    """Regression: 2026-05-18 event #1. StFuzS int32 cast turned image to all 0.

    If output image is mostly zero (blinding), sanity_nonzero must be low.
    """
    # Register a mock operator that simulates the old StFuzS bug (image -> 0)
    @mutation.register("StBlind", "state_out")
    def _blind(obs, ctx, cfg):
        img, nav = obs
        return [np.zeros_like(img), nav]

    _set_op("StBlind", intensity=1.0)
    obs = [np.ones((160, 80, 3), dtype=np.float32) * 0.5,
           np.array([0.1, 0.2, 0.3, 0.4, 0.5])]
    mutation.tick()
    mutation.state_out(obs)
    lines = _probe_lines(capsys.readouterr().out)
    assert len(lines) == 1
    m = re.search(r"sanity_nonzero=([\d.]+)", lines[0])
    assert m
    assert float(m.group(1)) < 0.05, (
        f"blinding bug must show sanity_nonzero ~ 0. got: {lines[0]}"
    )


# ----------- silent paths emit nothing ----------- #


def test_no_probe_when_mutation_none(capsys):
    reset_config(MutationConfig(mutation_type="none", seed=0, intensity=0.0))
    obs = [np.ones((160, 80, 3), dtype=np.float32),
           np.array([0.1, 0.2, 0.3, 0.4, 0.5])]
    mutation.tick()
    mutation.state_out(obs)
    mutation.action_in(np.array([0.0, 0.0]))
    mutation.reward_out(0.5)
    mutation.pv_out(np.array([0.0, 0.0]), 0.0, dist=None)
    mutation.es_sample(np.array([0.0, 0.0]), mean=None, cov_mat=None, dist=None)
    mutation.rc("cruise", 1.0)
    assert _probe_lines(capsys.readouterr().out) == []


def test_no_probe_when_op_not_in_category(capsys):
    """Op registered for action_in only; calling state_out falls through -> no probe."""
    @mutation.register("OnlyAction", "action_in")
    def _x(action, ctx, cfg):
        return action

    _set_op("OnlyAction", intensity=1.0)
    obs = [np.ones((160, 80, 3), dtype=np.float32),
           np.array([0.1, 0.2, 0.3, 0.4, 0.5])]
    mutation.tick()
    mutation.state_out(obs)  # no operator registered for state_out -> identity
    assert _probe_lines(capsys.readouterr().out) == []


# ----------- one-shot semantics ----------- #


def test_probe_only_emits_once_per_hook_op(capsys):
    _set_op("StFuzS", intensity=1.0)
    obs = [np.ones((160, 80, 3), dtype=np.float32) * 0.5,
           np.array([0.1, 0.2, 0.3, 0.4, 0.5])]
    for _ in range(20):
        mutation.tick()
        mutation.state_out(obs)
    lines = _probe_lines(capsys.readouterr().out)
    assert len(lines) == 1


def test_rc_dedupes_per_coef_name(capsys):
    """rc uses (hook, op, coef_name) as dedup key — each coef gets one line."""
    @mutation.register("RCMock", "rc")
    def _mock(coef_name, default, ctx, cfg):
        return default * 2.0

    _set_op("RCMock", intensity=1.0)
    # First touch of each coef name -> one line each
    mutation.rc("cruise", 1.0)
    mutation.rc("progress", 0.05)
    mutation.rc("cruise", 1.0)  # second touch of cruise -> no new line
    lines = _probe_lines(capsys.readouterr().out)
    assert len(lines) == 2
    assert any("coef=cruise" in ln for ln in lines)
    assert any("coef=progress" in ln for ln in lines)


def test_reset_clears_emission_state(capsys):
    _set_op("StFuzS", intensity=1.0)
    obs = [np.ones((160, 80, 3), dtype=np.float32) * 0.5,
           np.array([0.1, 0.2, 0.3, 0.4, 0.5])]
    mutation.tick()
    mutation.state_out(obs)
    capsys.readouterr()  # drain
    _probe.reset()
    mutation.state_out(obs)
    assert len(_probe_lines(capsys.readouterr().out)) == 1
