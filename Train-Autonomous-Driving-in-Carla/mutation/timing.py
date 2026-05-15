"""Timing dispatcher: P (periodical), R (random), S (sustained).

Operator name suffix encodes the timing:
  *P -> periodical: trigger if timestep > 0 and timestep % period_steps == 0
  *R -> random:     trigger if rng.random() < prob
  *S -> sustained:  always trigger

Default knobs (03-operators-spec.md):
  period_steps = 10
  prob         = 0.3
"""
DEFAULT_PERIOD_STEPS = 10
DEFAULT_PROB = 0.3


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
