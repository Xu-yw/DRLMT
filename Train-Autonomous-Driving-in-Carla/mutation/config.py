"""Mutation configuration. Loaded once from env vars; module-level singleton.

Env vars:
  MUTATION_TYPE       - operator name (e.g. StRepP, ESRemS); 'none' disables all hooks
  MUTATION_SEED       - int seed for mutation rng (independent of training/eval seeds)
  MUTATION_INTENSITY  - float intensity parameter, operator-specific meaning

Phase 2: only MUTATION_TYPE=none is wired (passthrough). Phase 3 will dispatch.
"""
import os
import random

import numpy as np


class MutationConfig:
    def __init__(self, mutation_type="none", seed=0, intensity=0.0):
        self.mutation_type = mutation_type
        self.seed = int(seed)
        self.intensity = float(intensity)
        self._rng = None
        self._np_rng = None
        if self.is_active:
            self._rng = random.Random(self.seed)
            self._np_rng = np.random.default_rng(self.seed)

    @classmethod
    def from_env(cls):
        return cls(
            mutation_type=os.environ.get("MUTATION_TYPE", "none"),
            seed=os.environ.get("MUTATION_SEED", "0"),
            intensity=os.environ.get("MUTATION_INTENSITY", "0.0"),
        )

    @property
    def is_active(self):
        return self.mutation_type != "none"

    @property
    def rng(self):
        return self._rng

    @property
    def np_rng(self):
        return self._np_rng

    def __repr__(self):
        return (
            f"MutationConfig(type={self.mutation_type!r}, "
            f"seed={self.seed}, intensity={self.intensity})"
        )


_config_singleton = None


def get_config():
    """Return module-level MutationConfig singleton (lazy-init from env)."""
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = MutationConfig.from_env()
    return _config_singleton


def reset_config(cfg=None):
    """Test helper. cfg=None forces re-read from env on next get_config()."""
    global _config_singleton
    _config_singleton = cfg
