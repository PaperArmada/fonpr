"""
FONPR simulation environment (S1).

API-identical to the live env: observation is a ``(samples, 3)`` float32
array of [throughput, large-on, small-on] rows (oldest first), actions are
``Discrete(3)`` (NOOP / go-large / go-small), reward is
``-(infra_cost + slo_penalty)`` in USD (S2.1).
"""

from __future__ import annotations

import logging
import typing
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fonpr.sim.config import SimConfig
from fonpr.sim.plant import PlantModel
from fonpr.sim.traffic import TrafficModel

logger = logging.getLogger(__name__)

ACTION_NOOP = 0
ACTION_LARGE = 1
ACTION_SMALL = 2


class FONPRSimEnv(gym.Env):
    """Offline, seedable stand-in for the live respons-nuances loop."""

    metadata: typing.ClassVar[dict] = {"render_modes": []}

    def __init__(self, config: SimConfig | None = None):
        super().__init__()
        self.config = config or SimConfig()
        samples = self.config.time.window_ticks
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(samples, 3), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._traffic: TrafficModel | None = None
        self._plant: PlantModel | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        cfg = self.config
        self._traffic = TrafficModel(cfg.traffic, cfg.time, self.np_random)
        self._plant = PlantModel(cfg.plant, cfg.econ, cfg.time)
        self._step_count = 0

        # Warm-up: fill the observation window before the episode starts.
        # Pre-episode cost and violations are not scored.
        offered = self._traffic.advance(cfg.time.window_ticks)
        result = self._plant.advance(offered)
        self._throughput_hist = result.served.copy()
        self._large_hist = result.large_on.copy()
        self._small_hist = result.small_on.copy()

        return (
            self._observation(),
            self._info(offered, result, violation_minutes=0.0, penalty_usd=0.0, applied=False),
        )

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._traffic is None or self._plant is None:
            raise RuntimeError("reset() must be called before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")
        cfg = self.config

        applied = False
        if action == ACTION_LARGE:
            applied = self._plant.request_transition("large")
        elif action == ACTION_SMALL:
            applied = self._plant.request_transition("small")

        offered = self._traffic.advance(cfg.time.ticks_per_step)
        result = self._plant.advance(offered)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(offered > 0, result.served / offered, 1.0)
        violation_ticks = int(np.sum(ratio < cfg.econ.slo_target))
        violation_minutes = violation_ticks * cfg.time.tick_minutes
        penalty = violation_minutes * cfg.econ.penalty_per_violation_minute_usd
        reward = -(result.infra_cost_usd + penalty)

        window = cfg.time.window_ticks
        self._throughput_hist = np.concatenate([self._throughput_hist, result.served])[-window:]
        self._large_hist = np.concatenate([self._large_hist, result.large_on])[-window:]
        self._small_hist = np.concatenate([self._small_hist, result.small_on])[-window:]

        self._step_count += 1
        truncated = self._step_count >= cfg.time.episode_steps

        return (
            self._observation(),
            float(reward),
            False,  # continuous task: never terminated (S1.4)
            truncated,
            self._info(
                offered,
                result,
                violation_minutes=violation_minutes,
                penalty_usd=penalty,
                applied=applied,
            ),
        )

    def _observation(self) -> np.ndarray:
        return np.stack(
            [self._throughput_hist, self._large_hist, self._small_hist], axis=1
        ).astype(np.float32)

    def _info(
        self,
        offered: np.ndarray,
        result,
        *,
        violation_minutes: float,
        penalty_usd: float,
        applied: bool,
    ) -> dict[str, Any]:
        return {
            "offered_load": float(offered.mean()),
            "served_load": float(result.served.mean()),
            "slo_violation": float(violation_minutes),
            "instance_type": self._plant.active_instance_type,
            "in_transition": self._plant.in_transition,
            "step_cost_usd": float(result.infra_cost_usd),
            "step_penalty_usd": float(penalty_usd),
            "offered_series": offered,
            "served_series": result.served,
            "capacity_series": result.capacity,
            "action_applied": applied,
        }
