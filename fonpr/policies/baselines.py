"""
Baseline policies (S3). The learned agent is judged only relative to these.

All baselines observe *served* throughput, which saturates at the active
capacity — the same partial observability the live system has. None of them
sees offered load directly.
"""

from __future__ import annotations

import numpy as np

from fonpr.policies.base import Policy, current_instance, observed_throughput
from fonpr.sim.config import SimConfig
from fonpr.sim.env import ACTION_LARGE, ACTION_NOOP, ACTION_SMALL


class NoopPolicy(Policy):
    """B0: never acts. Floor reference."""

    name = "noop"

    def act(self, obs: np.ndarray) -> int:
        return ACTION_NOOP


class ThresholdPolicy(Policy):
    """B1: V0-style hysteresis heuristic with patience and cooldown."""

    name = "threshold"

    def __init__(
        self,
        config: SimConfig,
        up_threshold: float = 0.8,
        down_threshold: float = 0.5,
        up_patience: int = 2,
        down_patience: int = 4,
        cooldown_steps: int = 4,
    ):
        self._small_cap = config.plant.small_capacity_bytes_per_sec
        self._up = up_threshold * self._small_cap
        self._down = down_threshold * self._small_cap
        self._up_patience = up_patience
        self._down_patience = down_patience
        self._cooldown_steps = cooldown_steps
        self.reset()

    def reset(self) -> None:
        self._up_count = 0
        self._down_count = 0
        self._cooldown = 0

    def act(self, obs: np.ndarray) -> int:
        if self._cooldown > 0:
            self._cooldown -= 1
            return ACTION_NOOP
        instance = current_instance(obs)
        if instance == "transition":
            return ACTION_NOOP
        tput = observed_throughput(obs)

        if instance == "small":
            self._down_count = 0
            self._up_count = self._up_count + 1 if tput > self._up else 0
            if self._up_count >= self._up_patience:
                self._up_count = 0
                self._cooldown = self._cooldown_steps
                return ACTION_LARGE
        else:
            self._up_count = 0
            self._down_count = self._down_count + 1 if tput < self._down else 0
            if self._down_count >= self._down_patience:
                self._down_count = 0
                self._cooldown = self._cooldown_steps
                return ACTION_SMALL
        return ACTION_NOOP


class ReactivePolicy(Policy):
    """B2: HPA-like target-utilization rule; single-step, no patience."""

    name = "reactive"

    def __init__(self, config: SimConfig, target_utilization: float = 0.7):
        self._small_cap = config.plant.small_capacity_bytes_per_sec
        self._large_cap = config.plant.large_capacity_bytes_per_sec
        self._target = target_utilization

    def act(self, obs: np.ndarray) -> int:
        instance = current_instance(obs)
        if instance == "transition":
            return ACTION_NOOP
        tput = observed_throughput(obs)
        if instance == "small" and tput > self._target * self._small_cap:
            return ACTION_LARGE
        if instance == "large" and tput < self._target * self._small_cap:
            return ACTION_SMALL
        return ACTION_NOOP


class ForecastPolicy(Policy):
    """B3: seasonal-naive forecast (same step yesterday), then size to fit.

    Chooses the cheapest instance whose capacity covers forecast x margin.
    Saturated history on the small instance forecasts at small capacity,
    and capacity x margin exceeds capacity, so a saturated yesterday
    correctly demands the large instance today.
    """

    name = "forecast"

    def __init__(self, config: SimConfig, safety_margin: float = 1.15):
        self._small_cap = config.plant.small_capacity_bytes_per_sec
        self._margin = safety_margin
        self._steps_per_day = int(24 * 60 / config.time.step_minutes)
        self.reset()

    def reset(self) -> None:
        self._history: list[float] = []

    def act(self, obs: np.ndarray) -> int:
        tput = observed_throughput(obs)
        self._history.append(tput)
        if len(self._history) > self._steps_per_day:
            forecast = self._history[-self._steps_per_day]
        else:
            forecast = tput
        needed = forecast * self._margin

        instance = current_instance(obs)
        if instance == "transition":
            return ACTION_NOOP
        if needed > self._small_cap:
            return ACTION_LARGE if instance == "small" else ACTION_NOOP
        return ACTION_SMALL if instance == "large" else ACTION_NOOP


def make_baselines(config: SimConfig) -> list[Policy]:
    """The standard baseline suite, in reporting order."""
    return [
        NoopPolicy(),
        ThresholdPolicy(config),
        ReactivePolicy(config),
        ForecastPolicy(config),
    ]
