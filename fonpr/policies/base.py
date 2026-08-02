"""
Policy seam (S3): observation -> action, nothing else.

Baselines, the learned agent, and the oracle all sit behind this interface
so the eval harness can swap them freely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Observation column indices (see FONPRSimEnv docstring). Columns 3-4 are
# the optional sin/cos time-of-day features (ADR-0002); they are appended,
# so the base indices hold in both variants.
COL_THROUGHPUT = 0
COL_LARGE_ON = 1
COL_SMALL_ON = 2
COL_SIN_TOD = 3
COL_COS_TOD = 4


class Policy(ABC):
    """A deployable decision rule over env observations."""

    name: str = "policy"

    def reset(self) -> None:  # noqa: B027 — stateless policies need no override
        """Clear any internal state before a new episode."""

    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
        """Map one observation to an action in {0: NOOP, 1: LARGE, 2: SMALL}."""


def current_instance(obs: np.ndarray) -> str:
    """Infer plant state from the newest observation row.

    Returns 'large', 'small', or 'transition' (both node groups billed).
    """
    large_on = obs[-1, COL_LARGE_ON] > 0.5
    small_on = obs[-1, COL_SMALL_ON] > 0.5
    if large_on and small_on:
        return "transition"
    return "large" if large_on else "small"


def observed_throughput(obs: np.ndarray) -> float:
    """Mean observed (served) throughput over the observation window, bytes/sec."""
    return float(obs[:, COL_THROUGHPUT].mean())
