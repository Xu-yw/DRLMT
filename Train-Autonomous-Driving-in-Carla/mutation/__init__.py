"""Mutation runtime."""
from .config import MutationConfig, get_config, reset_config
from .context import (
    OperatorContext,
    get_context,
    reset_episode,
    tick,
    current_step,
    reset_all,
)
from .hooks import state_out, action_in, reward_out, rc, pv_out, es_sample
from .registry import register, get as get_operator, list_operators, reset_registry
from .timing import trigger

# Triggers @register side effects for all bundled operators (Phase 3.1+).
# Must be imported AFTER registry / config / hooks are defined.
from . import operators  # noqa: E402,F401

# Snapshot the bundled-operator registry as baseline so test fixtures
# can wipe ad-hoc registrations without losing Phase 3.1+ operators.
from .registry import _capture_baseline as _capture_baseline  # noqa: E402
_capture_baseline()

__all__ = [
    "MutationConfig",
    "get_config",
    "reset_config",
    "OperatorContext",
    "get_context",
    "reset_episode",
    "tick",
    "current_step",
    "reset_all",
    "state_out",
    "action_in",
    "reward_out",
    "rc",
    "pv_out",
    "es_sample",
    "register",
    "get_operator",
    "list_operators",
    "reset_registry",
    "trigger",
]
