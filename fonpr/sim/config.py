"""
Simulation configuration (S1.5).

Canonical config pattern for this codebase: frozen dataclasses with a strict
YAML loader. Unknown keys are errors, never silently ignored.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TimeConfig:
    """Sim time model (S1.4). One env step = ``step_minutes`` of sim time."""

    step_minutes: int = 15
    window_minutes: int = 15
    sample_rate_per_minute: int = 1
    episode_days: float = 7.0

    def __post_init__(self) -> None:
        if self.step_minutes <= 0 or self.window_minutes <= 0:
            raise ValueError("step_minutes and window_minutes must be positive")
        if self.sample_rate_per_minute < 1:
            raise ValueError("sample_rate_per_minute must be >= 1")
        if self.episode_days <= 0:
            raise ValueError("episode_days must be positive")

    @property
    def tick_minutes(self) -> float:
        """Duration of one internal sample tick, in minutes."""
        return 1.0 / self.sample_rate_per_minute

    @property
    def ticks_per_step(self) -> int:
        return self.step_minutes * self.sample_rate_per_minute

    @property
    def window_ticks(self) -> int:
        """Number of samples in one observation (spec: ``samples``)."""
        return self.window_minutes * self.sample_rate_per_minute

    @property
    def episode_steps(self) -> int:
        return int(self.episode_days * 24 * 60 / self.step_minutes)


@dataclass(frozen=True)
class TrafficConfig:
    """Offered-load generator parameters (S1.2). Loads are bytes/second."""

    base_load_bytes_per_sec: float = 12.5e6  # 100 Mbit/s
    diurnal_amplitude: float = 0.6  # peak = base * (1 + amplitude)
    diurnal_period_hours: float = 24.0
    diurnal_phase_hours: float = 0.0
    noise_sigma_frac: float = 0.05
    noise_ar1_rho: float = 0.5
    burst_rate_per_day: float = 2.0
    burst_magnitude_log_mu: float = 0.9  # lognormal multiplier, median ~2.5x
    burst_magnitude_log_sigma: float = 0.25
    burst_duration_minutes_min: float = 10.0
    burst_duration_minutes_max: float = 45.0
    drift_frac_per_day: float = 0.0

    def __post_init__(self) -> None:
        if self.base_load_bytes_per_sec <= 0:
            raise ValueError("base_load_bytes_per_sec must be positive")
        if not 0 <= self.diurnal_amplitude < 1:
            raise ValueError("diurnal_amplitude must be in [0, 1)")
        if not 0 <= self.noise_ar1_rho < 1:
            raise ValueError("noise_ar1_rho must be in [0, 1)")
        if self.burst_duration_minutes_min > self.burst_duration_minutes_max:
            raise ValueError("burst duration min must be <= max")

    @property
    def peak_diurnal_load(self) -> float:
        return self.base_load_bytes_per_sec * (1 + self.diurnal_amplitude)


@dataclass(frozen=True)
class PlantConfig:
    """Cluster response model (S1.3). Capacities calibrated per ADR-0001/D7."""

    large_instance_type: str = "m4.xlarge"
    small_instance_type: str = "t3.medium"
    large_capacity_bytes_per_sec: float = 24.0e6  # 1.2x default peak diurnal
    small_capacity_bytes_per_sec: float = 8.0e6  # 0.4x default peak diurnal
    transition_lag_minutes: int = 5
    initial_instance: str = "small"

    def __post_init__(self) -> None:
        if self.initial_instance not in ("small", "large"):
            raise ValueError("initial_instance must be 'small' or 'large'")
        if self.large_capacity_bytes_per_sec <= self.small_capacity_bytes_per_sec:
            raise ValueError("large capacity must exceed small capacity")
        if self.transition_lag_minutes < 0:
            raise ValueError("transition_lag_minutes must be >= 0")

    def capacity(self, instance: str) -> float:
        if instance == "large":
            return self.large_capacity_bytes_per_sec
        if instance == "small":
            return self.small_capacity_bytes_per_sec
        raise ValueError(f"unknown instance {instance!r}")


@dataclass(frozen=True)
class EconConfig:
    """Cost and SLO-penalty model (S2). All values in USD."""

    hourly_cost_usd: dict[str, float] = field(
        default_factory=lambda: {"m4.xlarge": 0.20, "t3.medium": 0.0416}
    )
    slo_target: float = 0.995  # ADR-0001/D5
    slo_penalty_multiplier: float = 20.0  # x most-expensive hourly cost, per violation-minute

    def __post_init__(self) -> None:
        if not 0 < self.slo_target <= 1:
            raise ValueError("slo_target must be in (0, 1]")
        if self.slo_penalty_multiplier < 0:
            raise ValueError("slo_penalty_multiplier must be >= 0")

    def hourly_cost(self, instance_type: str) -> float:
        if instance_type not in self.hourly_cost_usd:
            raise KeyError(f"instance type {instance_type!r} not in pricing table")
        return self.hourly_cost_usd[instance_type]

    @property
    def penalty_per_violation_minute_usd(self) -> float:
        return self.slo_penalty_multiplier * max(self.hourly_cost_usd.values()) / 60.0


@dataclass(frozen=True)
class SimConfig:
    """Top-level simulator configuration. Load from YAML via ``from_yaml``."""

    time: TimeConfig = field(default_factory=TimeConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    plant: PlantConfig = field(default_factory=PlantConfig)
    econ: EconConfig = field(default_factory=EconConfig)

    def __post_init__(self) -> None:
        for instance_type in (self.plant.large_instance_type, self.plant.small_instance_type):
            self.econ.hourly_cost(instance_type)  # fail fast on missing pricing

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SimConfig:
        return _dataclass_from_dict(cls, raw, path="sim")

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimConfig:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: top level of a sim config must be a mapping")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def _dataclass_from_dict(cls: type, raw: dict[str, Any], path: str) -> Any:
    """Construct a (possibly nested) dataclass from a dict, rejecting unknown keys."""
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise KeyError(f"{path}: unknown config key(s) {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        f = known[name]
        if dataclasses.is_dataclass(f.type) or (
            isinstance(f.type, str) and f.type in _NESTED_TYPES
        ):
            nested_cls = _NESTED_TYPES[f.type] if isinstance(f.type, str) else f.type
            if not isinstance(value, dict):
                raise TypeError(f"{path}.{name}: expected a mapping")
            kwargs[name] = _dataclass_from_dict(nested_cls, value, path=f"{path}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


# `from __future__ import annotations` stringifies field types; map them back.
_NESTED_TYPES: dict[str, type] = {
    "TimeConfig": TimeConfig,
    "TrafficConfig": TrafficConfig,
    "PlantConfig": PlantConfig,
    "EconConfig": EconConfig,
}
