"""
Offered-load generator (S1.2).

Composes diurnal, AR(1) noise, Poisson burst, and drift components. All
randomness flows from the ``numpy.random.Generator`` passed in by the env;
same generator state + same config => identical output.
"""

from __future__ import annotations

import numpy as np

from fonpr.sim.config import TimeConfig, TrafficConfig


class TrafficModel:
    """Stateful per-tick offered-load generator, in bytes/second."""

    def __init__(self, cfg: TrafficConfig, time_cfg: TimeConfig, rng: np.random.Generator):
        self._cfg = cfg
        self._tick_minutes = time_cfg.tick_minutes
        self._rng = rng
        self.reset()

    def reset(self) -> None:
        self._tick_index = 0
        self._noise_state = 0.0
        self._burst_ticks_remaining = 0
        self._burst_multiplier = 1.0

    def advance(self, n_ticks: int) -> np.ndarray:
        """Generate offered load for the next ``n_ticks`` samples."""
        cfg = self._cfg
        # Per-tick burst start probability from the daily Poisson rate.
        burst_p = cfg.burst_rate_per_day * self._tick_minutes / (24.0 * 60.0)
        rho = cfg.noise_ar1_rho
        innovation_scale = cfg.noise_sigma_frac * np.sqrt(1.0 - rho**2)

        # Deterministic components, vectorized.
        minutes = (self._tick_index + np.arange(n_ticks)) * self._tick_minutes
        hours = minutes / 60.0
        diurnal = 1.0 + cfg.diurnal_amplitude * np.sin(
            2.0 * np.pi * (hours - cfg.diurnal_phase_hours) / cfg.diurnal_period_hours
        )
        drift = 1.0 + cfg.drift_frac_per_day * (minutes / (24.0 * 60.0))

        # Stochastic components: batch the per-tick draws, keep the (cheap)
        # sequential state machines as scalar-float loops.
        innovations = self._rng.normal(0.0, innovation_scale, size=n_ticks)
        start_draws = self._rng.random(size=n_ticks)
        noise = np.empty(n_ticks, dtype=np.float64)
        burst = np.empty(n_ticks, dtype=np.float64)
        noise_state = self._noise_state
        for i in range(n_ticks):
            noise_state = rho * noise_state + innovations[i]
            noise[i] = max(1.0 + noise_state, 0.1)

            if self._burst_ticks_remaining == 0 and start_draws[i] < burst_p:
                duration_minutes = self._rng.uniform(
                    cfg.burst_duration_minutes_min, cfg.burst_duration_minutes_max
                )
                self._burst_ticks_remaining = max(
                    round(duration_minutes / self._tick_minutes), 1
                )
                self._burst_multiplier = max(
                    self._rng.lognormal(
                        cfg.burst_magnitude_log_mu, cfg.burst_magnitude_log_sigma
                    ),
                    1.0,
                )
            if self._burst_ticks_remaining > 0:
                burst[i] = self._burst_multiplier
                self._burst_ticks_remaining -= 1
            else:
                burst[i] = 1.0
        self._noise_state = noise_state

        self._tick_index += n_ticks
        return np.maximum(
            cfg.base_load_bytes_per_sec * diurnal * drift * noise * burst, 0.0
        )
