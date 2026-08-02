"""
Hindsight-optimal oracle (S3): dynamic programming over the full offered-load
trace. Upper bound for regret computation only — never a deployable policy,
because it sees the future.
"""

from __future__ import annotations

import numpy as np

from fonpr.policies.base import Policy
from fonpr.sim.config import SimConfig
from fonpr.sim.env import ACTION_LARGE, ACTION_NOOP, ACTION_SMALL

_STATES = ("small", "large")


def _step_cost(
    config: SimConfig, state: str, target: str, offered: np.ndarray
) -> float:
    """Infra cost + SLO penalty for one step starting in ``state``.

    ``target`` == ``state`` models NOOP; otherwise a transition starts at the
    step boundary (lag < step length, so it always completes in-step —
    matching the plant model).
    """
    plant, econ, time_cfg = config.plant, config.econ, config.time
    tick_hours = time_cfg.tick_minutes / 60.0
    n = len(offered)
    capacity = np.empty(n)

    if target == state:
        capacity[:] = plant.capacity(state)
        infra = econ.hourly_cost(
            plant.large_instance_type if state == "large" else plant.small_instance_type
        ) * tick_hours * n
    else:
        lag = min(round(plant.transition_lag_minutes / time_cfg.tick_minutes), n)
        capacity[:lag] = plant.small_capacity_bytes_per_sec
        capacity[lag:] = plant.capacity(target)
        both = (
            econ.hourly_cost(plant.large_instance_type)
            + econ.hourly_cost(plant.small_instance_type)
        )
        after = econ.hourly_cost(
            plant.large_instance_type if target == "large" else plant.small_instance_type
        )
        infra = both * tick_hours * lag + after * tick_hours * (n - lag)

    served = np.minimum(offered, capacity)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(offered > 0, served / offered, 1.0)
    violation_minutes = float(np.sum(ratio < econ.slo_target)) * time_cfg.tick_minutes
    return infra + violation_minutes * econ.penalty_per_violation_minute_usd


def plan_oracle_actions(
    offered_steps: list[np.ndarray], config: SimConfig, initial_state: str | None = None
) -> tuple[list[int], float]:
    """Backward-DP over per-step offered-load arrays.

    Returns (actions, optimal_total_cost) starting from ``initial_state``
    (defaults to the plant's configured initial instance).
    """
    start = initial_state or config.plant.initial_instance
    n_steps = len(offered_steps)
    # value[s] = minimal cost-to-go from state s at the current step boundary.
    value = {s: 0.0 for s in _STATES}
    best_target: list[dict[str, str]] = [{} for _ in range(n_steps)]

    for t in range(n_steps - 1, -1, -1):
        new_value = {}
        for state in _STATES:
            best = None
            for target in _STATES:
                cost = _step_cost(config, state, target, offered_steps[t]) + value[target]
                if best is None or cost < best[0]:
                    best = (cost, target)
            new_value[state] = best[0]
            best_target[t][state] = best[1]
        value = new_value

    actions: list[int] = []
    state = start
    for t in range(n_steps):
        target = best_target[t][state]
        if target == state:
            actions.append(ACTION_NOOP)
        else:
            actions.append(ACTION_LARGE if target == "large" else ACTION_SMALL)
        state = target
    return actions, value[start]


class OraclePolicy(Policy):
    """Replays a precomputed hindsight-optimal action plan."""

    name = "oracle"

    def __init__(self, actions: list[int]):
        self._actions = actions
        self.reset()

    def reset(self) -> None:
        self._t = 0

    def act(self, obs: np.ndarray) -> int:
        if self._t >= len(self._actions):
            return ACTION_NOOP
        action = self._actions[self._t]
        self._t += 1
        return action


__all__ = ["OraclePolicy", "plan_oracle_actions"]
