"""Mutation runtime probe — first-call evidence emission (P0-1).

Emits one [MUT-PROBE] log line per (hook, op) pair when the dispatcher first
routes to a registered operator. Lets a wrapper / human reader fail-fast on
misconfigured mutation runs.

Two bug classes this guards against:

  1. "wired but silent" — cfg.intensity=0 -> noise std=0 -> input==output
     (root cause of 2026-05-18 event #2: watchdog didn't pass MUTATION_INTENSITY)
  2. "wired but destroyed" — dtype / clip error blanks the signal
     (root cause of 2026-05-18 event #1: StFuzS int32 cast turned image to 0)

Output line schema (single line, space-separated key=value):

  [MUT-PROBE] hook=<state_out|action_in|reward_out|pv_out|es_sample|rc>
              op=<op_name> intensity=<float>
              perturbed=<True|False>
              delta_mean=<float>
              [hook-specific extras...]
              step=<int>

A `perturbed=False` line is the fail-fast signal — operator was reached but did
not actually change its input. Combined with `delta_mean=0.000000` this
unambiguously identifies the intensity=0 / no-op class.

`sanity_nonzero` on the state_out probe guards the StFuzS-blinding regression:
if the output image is mostly zeros, sanity_nonzero will be near 0 even though
perturbed=True.
"""
import sys

import numpy as np


_emitted = set()  # keys: (hook, op) tuples (rc uses (hook, op, coef_name))


def reset():
    """Test helper: clear emission state so the next call re-emits."""
    _emitted.clear()


def _to_array(x):
    """Best-effort: torch.Tensor / list / ndarray / scalar -> np.ndarray float64.

    Returns None on unsupported input. Detaches torch tensors and moves to CPU.
    """
    try:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float64)
    except Exception:
        return None


def _arr_delta(a, b):
    """mean(|a - b|) over matching-shape arrays; NaN if shapes mismatch."""
    aa = _to_array(a)
    bb = _to_array(b)
    if aa is None or bb is None or aa.shape != bb.shape:
        return float("nan")
    return float(np.mean(np.abs(aa - bb)))


def _nonzero_ratio(img):
    """Fraction of strictly-positive pixels (guards StFuzS-blinding bug)."""
    arr = _to_array(img)
    if arr is None or arr.size == 0:
        return float("nan")
    return float((arr > 0).sum() / arr.size)


def _safe_max(*vals):
    """max() over non-NaN values; NaN if all are NaN."""
    valid = [v for v in vals if v == v]  # NaN != NaN
    return max(valid) if valid else float("nan")


def _is_perturbed(delta):
    return delta == delta and delta > 1e-8  # NaN-safe


def _emit(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def emit_state_out(op, intensity, obs_in, obs_out, step):
    key = ("state_out", op)
    if key in _emitted:
        return
    _emitted.add(key)
    img_delta = nav_delta = sanity_nonzero = float("nan")
    try:
        img_in, nav_in = obs_in
        img_out, nav_out = obs_out
        img_delta = _arr_delta(img_in, img_out)
        nav_delta = _arr_delta(nav_in, nav_out)
        sanity_nonzero = _nonzero_ratio(img_out)
    except Exception:
        pass
    delta_mean = _safe_max(img_delta, nav_delta)
    _emit(
        f"[MUT-PROBE] hook=state_out op={op} intensity={intensity:.3f} "
        f"perturbed={_is_perturbed(delta_mean)} delta_mean={delta_mean:.6f} "
        f"img_delta={img_delta:.6f} nav_delta={nav_delta:.6f} "
        f"sanity_nonzero={sanity_nonzero:.4f} step={step}"
    )


def emit_action_in(op, intensity, act_in, act_out, step):
    key = ("action_in", op)
    if key in _emitted:
        return
    _emitted.add(key)
    delta_mean = _arr_delta(act_in, act_out)
    _emit(
        f"[MUT-PROBE] hook=action_in op={op} intensity={intensity:.3f} "
        f"perturbed={_is_perturbed(delta_mean)} delta_mean={delta_mean:.6f} step={step}"
    )


def emit_reward_out(op, intensity, r_in, r_out, step):
    key = ("reward_out", op)
    if key in _emitted:
        return
    _emitted.add(key)
    try:
        delta_mean = abs(float(r_in) - float(r_out))
    except Exception:
        delta_mean = float("nan")
    _emit(
        f"[MUT-PROBE] hook=reward_out op={op} intensity={intensity:.3f} "
        f"perturbed={_is_perturbed(delta_mean)} delta_mean={delta_mean:.6f} step={step}"
    )


def emit_pv_out(op, intensity, act_in, act_out, lp_in, lp_out, step):
    key = ("pv_out", op)
    if key in _emitted:
        return
    _emitted.add(key)
    a_delta = _arr_delta(act_in, act_out)
    try:
        lp_in_arr = _to_array(lp_in)
        lp_out_arr = _to_array(lp_out)
        lp_delta = float("nan")
        if lp_in_arr is not None and lp_out_arr is not None:
            lp_delta = float(np.mean(np.abs(lp_in_arr - lp_out_arr)))
    except Exception:
        lp_delta = float("nan")
    _emit(
        f"[MUT-PROBE] hook=pv_out op={op} intensity={intensity:.3f} "
        f"perturbed={_is_perturbed(a_delta)} delta_mean={a_delta:.6f} "
        f"logprob_delta={lp_delta:.6f} step={step}"
    )


def emit_es_sample(op, intensity, sample_in, sample_out, step):
    key = ("es_sample", op)
    if key in _emitted:
        return
    _emitted.add(key)
    delta_mean = _arr_delta(sample_in, sample_out)
    _emit(
        f"[MUT-PROBE] hook=es_sample op={op} intensity={intensity:.3f} "
        f"perturbed={_is_perturbed(delta_mean)} delta_mean={delta_mean:.6f} step={step}"
    )


def emit_rc(op, intensity, coef_name, default, returned, step):
    key = ("rc", op, coef_name)
    if key in _emitted:
        return
    _emitted.add(key)
    try:
        delta_mean = abs(float(default) - float(returned))
    except Exception:
        delta_mean = float("nan")
    _emit(
        f"[MUT-PROBE] hook=rc op={op} coef={coef_name} intensity={intensity:.3f} "
        f"perturbed={_is_perturbed(delta_mean)} delta_mean={delta_mean:.6f} step={step}"
    )
