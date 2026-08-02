"""
Plant models: binary two-tier sizing (S1.3) and the homogeneous node pool
(S14, ADR-0004).

Both capture the three economics-bearing behaviors of the real system:
saturation (offered load above capacity is dropped), transition lag
(capacity held at the pre-transition level while a resize completes), and
transition co-billing (booting/draining nodes accrue cost during the lag).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fonpr.sim.config import EconConfig, PlantConfig, PoolConfig, TimeConfig


@dataclass
class PlantStepResult:
    served: np.ndarray  # bytes/sec per tick
    capacity: np.ndarray  # bytes/sec per tick
    large_on: np.ndarray  # 0/1 per tick (billed this tick)
    small_on: np.ndarray  # 0/1 per tick (billed this tick)
    serving_large: np.ndarray  # 0/1 per tick: the node carrying traffic is large
    infra_cost_usd: float  # price-model cost; the energy variant recomputes (S13)


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
        serving_large = np.zeros(n, dtype=np.float64)
        cost = 0.0
        tick_hours = self._tick_minutes / 60.0
        cfg, econ = self._cfg, self._econ

        # Price-model bookkeeping is meaningless (and its table optional)
        # under the energy model; the env recomputes cost there (S13).
        def hourly(instance_type: str) -> float:
            return 0.0 if econ.power is not None else econ.hourly_cost(instance_type)

        i = 0
        while i < n:
            if self.in_transition:
                # Degraded to the smaller capacity; both node groups billed.
                # The pre-transition node keeps carrying the traffic.
                k = min(self._transition_ticks_remaining, n - i)
                segment = slice(i, i + k)
                capacity[segment] = cfg.small_capacity_bytes_per_sec
                large_on[segment] = 1.0
                small_on[segment] = 1.0
                serving_large[segment] = 1.0 if self.active == "large" else 0.0
                cost += (
                    hourly(cfg.large_instance_type) + hourly(cfg.small_instance_type)
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
                serving_large[segment] = 1.0 if self.active == "large" else 0.0
                cost += hourly(self.active_instance_type) * tick_hours * (n - i)
                i = n

        return PlantStepResult(
            served=np.minimum(offered, capacity),
            capacity=capacity,
            large_on=large_on,
            small_on=small_on,
            serving_large=serving_large,
            infra_cost_usd=cost,
        )


@dataclass
class PoolStepResult:
    served: np.ndarray  # bytes/sec per tick
    capacity: np.ndarray  # bytes/sec per tick
    active_count: np.ndarray  # nodes serving traffic, per tick
    billed_count: np.ndarray  # nodes accruing cost, per tick (>= active during a resize)
    target_count: np.ndarray  # resize destination per tick (== active outside a resize)
    infra_cost_usd: float  # price-model cost; the energy variant recomputes (S13)


class PoolPlantModel:
    """Homogeneous node pool (S14): actions set the target node count.

    Resize n -> m: for the lag, capacity stays at the old count (booting
    nodes are not ready; draining nodes still serve) while max(n, m) nodes
    bill (booting nodes bill from launch; draining nodes until drained).
    Requests during a resize are ignored.
    """

    def __init__(self, cfg: PoolConfig, econ: EconConfig, time_cfg: TimeConfig):
        self._cfg = cfg
        self._econ = econ
        self._tick_minutes = time_cfg.tick_minutes
        self.reset()

    def reset(self) -> None:
        self.active: int = self._cfg.initial_nodes
        self.target: int | None = None  # non-None while a resize is in flight
        self._transition_ticks_remaining = 0

    @property
    def in_transition(self) -> bool:
        return self.target is not None

    def request_target(self, count: int) -> bool:
        """Start a resize to ``count`` nodes; True if plant state changed."""
        if not self._cfg.min_nodes <= count <= self._cfg.max_nodes:
            raise ValueError(
                f"count {count!r} outside [{self._cfg.min_nodes}, {self._cfg.max_nodes}]"
            )
        if self.in_transition or count == self.active:
            return False
        self.target = count
        lag_ticks = round(self._cfg.transition_lag_minutes / self._tick_minutes)
        self._transition_ticks_remaining = lag_ticks
        if lag_ticks == 0:
            self.active = count
            self.target = None
        return True

    def advance(self, offered: np.ndarray) -> PoolStepResult:
        """Advance one tick per element of ``offered``; segment-filled like PlantModel."""
        n = len(offered)
        capacity = np.empty(n, dtype=np.float64)
        active_count = np.empty(n, dtype=np.float64)
        billed_count = np.empty(n, dtype=np.float64)
        target_count = np.empty(n, dtype=np.float64)
        cost = 0.0
        tick_hours = self._tick_minutes / 60.0
        cfg, econ = self._cfg, self._econ

        # Price bookkeeping is unused (and its table optional) under the
        # energy model; the env recomputes cost there (S13).
        node_hourly = 0.0 if econ.power is not None else econ.hourly_cost(cfg.node_type)

        i = 0
        while i < n:
            if self.in_transition:
                k = min(self._transition_ticks_remaining, n - i)
                segment = slice(i, i + k)
                billed = max(self.active, self.target)  # type: ignore[type-var]
                capacity[segment] = self.active * cfg.node_capacity_bytes_per_sec
                active_count[segment] = self.active
                billed_count[segment] = billed
                target_count[segment] = self.target
                cost += node_hourly * billed * tick_hours * k
                self._transition_ticks_remaining -= k
                if self._transition_ticks_remaining <= 0:
                    self.active = self.target  # type: ignore[assignment]
                    self.target = None
                i += k
            else:
                segment = slice(i, n)
                capacity[segment] = self.active * cfg.node_capacity_bytes_per_sec
                active_count[segment] = self.active
                billed_count[segment] = self.active
                target_count[segment] = self.active
                cost += node_hourly * self.active * tick_hours * (n - i)
                i = n

        return PoolStepResult(
            served=np.minimum(offered, capacity),
            capacity=capacity,
            active_count=active_count,
            billed_count=billed_count,
            target_count=target_count,
            infra_cost_usd=cost,
        )
