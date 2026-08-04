"""
B4 — model-predictive control baseline (S3; the ADR-0005 rung-2 competitor).

The strongest deployable non-RL competitor: B3's seasonal-naive forecaster
feeding the oracle's DP over the twin's own cost model, on a receding
horizon. B4 shares B3's forecaster and safety margin by construction, so
B4 minus B3 isolates the value of transition-aware multi-step scheduling,
and oracle minus B4 isolates the value of perfect foresight.

Honestly deployable: it observes served throughput only (the same partial
observability every baseline has), forecasts only from that history, and
prices candidate plans with the same single-sourced costing the plant
bills with (``fonpr/sim/costing.py``). Given exact forecasts, unit margin,
and a horizon covering the remaining episode, the receding-horizon plan
reproduces the oracle exactly (principle of optimality; pinned by test).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from fonpr.policies.base import (
    Policy,
    current_instance,
    current_node_count,
    observed_throughput,
    pool_in_transition,
)
from fonpr.policies.oracle import pool_step_cost, step_cost
from fonpr.sim.config import PoolConfig, SimConfig
from fonpr.sim.env import ACTION_LARGE, ACTION_NOOP, ACTION_SMALL


def plan_first_target(
    states: Sequence,
    start,
    levels: Sequence[float],
    cost_of: Callable[[object, object, float], float],
):
    """One receding-horizon DP pass; returns the target for the upcoming step.

    ``levels`` holds one forecast offered-load level per horizon step;
    ``cost_of(state, target, level)`` prices one step. Backward induction
    with terminal value zero, iterating states and targets in the given
    order with strict improvement — the same tie-breaking as the oracle DP,
    which the exact-forecast equivalence test relies on.
    """
    value = dict.fromkeys(states, 0.0)
    first = start
    for t in range(len(levels) - 1, -1, -1):
        level = levels[t]
        new_value = {}
        for state in states:
            best_cost = None
            best_target = state
            for target in states:
                total = cost_of(state, target, level) + value[target]
                if best_cost is None or total < best_cost:
                    best_cost, best_target = total, target
            new_value[state] = best_cost
            if t == 0 and state == start:
                first = best_target
        value = new_value
    return first


class _MpcBase(Policy):
    """Shared forecaster, cost cache, and planning loop for both variants."""

    name = "mpc"

    def __init__(
        self, config: SimConfig, safety_margin: float = 1.15, horizon_hours: float = 24.0
    ):
        self._config = config
        self._margin = safety_margin
        step = config.time.step_minutes
        self._horizon_steps = max(1, round(horizon_hours * 60 / step))
        self._steps_per_day = int(24 * 60 / step)
        self._ticks_per_step = config.time.ticks_per_step
        self.reset()

    def reset(self) -> None:
        self._history: list[float] = []
        # (state, target, level) -> step cost. Levels recur (seasonal
        # forecasts repeat observed values), so planning is amortized to a
        # handful of fresh costings per step.
        self._cost_cache: dict = {}

    def _forecast_levels(self, tput: float) -> list[float]:
        """B3's seasonal-naive forecast extended over the horizon.

        Future step m forecasts the observation from exactly one day
        earlier (index n + m - steps_per_day, where the upcoming step is
        B3's ``history[-steps_per_day]``); persistence of the current
        observation fills in wherever a full day of history is absent.
        """
        n = len(self._history)
        levels = []
        for m in range(self._horizon_steps):
            j = n + m - self._steps_per_day
            base = self._history[j] if 0 <= j < n else tput
            levels.append(base * self._margin)
        return levels

    def _cost_of(self, state, target, level: float) -> float:
        key = (state, target, level)
        cost = self._cost_cache.get(key)
        if cost is None:
            offered = np.full(self._ticks_per_step, level)
            cost = self._step_cost(state, target, offered)
            self._cost_cache[key] = cost
        return cost

    def _step_cost(self, state, target, offered: np.ndarray) -> float:
        raise NotImplementedError


class MpcPolicy(_MpcBase):
    """B4 (binary plant): receding-horizon DP over the small/large states."""

    _STATES = ("small", "large")

    def _step_cost(self, state, target, offered: np.ndarray) -> float:
        return step_cost(self._config, state, target, offered)

    def act(self, obs: np.ndarray) -> int:
        tput = observed_throughput(obs)
        self._history.append(tput)
        state = current_instance(obs)
        if state == "transition":
            return ACTION_NOOP
        target = plan_first_target(
            self._STATES, state, self._forecast_levels(tput), self._cost_of
        )
        if target == state:
            return ACTION_NOOP
        return ACTION_LARGE if target == "large" else ACTION_SMALL


class MpcPoolPolicy(_MpcBase):
    """B4 (pool plant, S14): receding-horizon DP over the node-count states."""

    def __init__(
        self, config: SimConfig, safety_margin: float = 1.15, horizon_hours: float = 24.0
    ):
        super().__init__(config, safety_margin, horizon_hours)
        self._pool: PoolConfig = config.plant.pool
        self._states = tuple(range(self._pool.min_nodes, self._pool.max_nodes + 1))

    def _step_cost(self, state, target, offered: np.ndarray) -> float:
        return pool_step_cost(self._config, state, target, offered)

    def act(self, obs: np.ndarray) -> int:
        pool = self._pool
        tput = observed_throughput(obs)
        self._history.append(tput)
        count = current_node_count(obs, pool.max_nodes)
        if pool_in_transition(obs, pool.max_nodes):
            return pool.action_of_count(count)
        target = plan_first_target(
            self._states, count, self._forecast_levels(tput), self._cost_of
        )
        return pool.action_of_count(target)


__all__ = ["MpcPolicy", "MpcPoolPolicy", "plan_first_target"]
