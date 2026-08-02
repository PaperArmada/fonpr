"""
Plant model (S1.3): how the cluster responds to sizing actions.

Captures the three economics-bearing behaviors of the real system:
saturation (offered load above capacity is dropped), transition lag
(capacity degraded to the smaller instance while a transition completes),
and transition double-billing (both node groups accrue cost while a
transition is in flight).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fonpr.sim.config import EconConfig, PlantConfig, TimeConfig


@dataclass
class PlantStepResult:
    served: np.ndarray  # bytes/sec per tick
    capacity: np.ndarray  # bytes/sec per tick
    large_on: np.ndarray  # 0/1 per tick (billed this tick)
    small_on: np.ndarray  # 0/1 per tick (billed this tick)
    infra_cost_usd: float


class PlantModel:
    """Stateful instance-sizing plant. Actions apply at step boundaries."""

    def __init__(self, cfg: PlantConfig, econ: EconConfig, time_cfg: TimeConfig):
        self._cfg = cfg
        self._econ = econ
        self._tick_minutes = time_cfg.tick_minutes
        self.reset()

    def reset(self) -> None:
        self.active: str = self._cfg.initial_instance
        self.target: str | None = None  # non-None while a transition is in flight
        self._transition_ticks_remaining = 0

    @property
    def in_transition(self) -> bool:
        return self.target is not None

    @property
    def active_instance_type(self) -> str:
        return (
            self._cfg.large_instance_type
            if self.active == "large"
            else self._cfg.small_instance_type
        )

    def request_transition(self, target: str) -> bool:
        """Start a transition to ``target`` ('large'|'small').

        Returns True if the request changed plant state. Requests are ignored
        (returning False) while a transition is already in flight or when the
        target is already active.
        """
        if target not in ("large", "small"):
            raise ValueError(f"unknown target {target!r}")
        if self.in_transition or target == self.active:
            return False
        self.target = target
        lag_ticks = round(self._cfg.transition_lag_minutes / self._tick_minutes)
        self._transition_ticks_remaining = lag_ticks
        if lag_ticks == 0:
            self.active = target
            self.target = None
        return True

    def advance(self, offered: np.ndarray) -> PlantStepResult:
        """Advance one tick per element of ``offered``; return served load and billing.

        Capacity is piecewise-constant (a transition segment, then steady
        state), so billing and capacity are filled per segment rather than
        per tick.
        """
        n = len(offered)
        capacity = np.empty(n, dtype=np.float64)
        large_on = np.zeros(n, dtype=np.float64)
        small_on = np.zeros(n, dtype=np.float64)
        cost = 0.0
        tick_hours = self._tick_minutes / 60.0
        cfg, econ = self._cfg, self._econ

        i = 0
        while i < n:
            if self.in_transition:
                # Degraded to the smaller capacity; both node groups billed.
                k = min(self._transition_ticks_remaining, n - i)
                segment = slice(i, i + k)
                capacity[segment] = cfg.small_capacity_bytes_per_sec
                large_on[segment] = 1.0
                small_on[segment] = 1.0
                cost += (
                    econ.hourly_cost(cfg.large_instance_type)
                    + econ.hourly_cost(cfg.small_instance_type)
                ) * tick_hours * k
                self._transition_ticks_remaining -= k
                if self._transition_ticks_remaining <= 0:
                    self.active = self.target  # type: ignore[assignment]
                    self.target = None
                i += k
            else:
                segment = slice(i, n)
                capacity[segment] = cfg.capacity(self.active)
                (large_on if self.active == "large" else small_on)[segment] = 1.0
                cost += econ.hourly_cost(self.active_instance_type) * tick_hours * (n - i)
                i = n

        return PlantStepResult(
            served=np.minimum(offered, capacity),
            capacity=capacity,
            large_on=large_on,
            small_on=small_on,
            infra_cost_usd=cost,
        )
