"""
Shared step-costing (S2 price model / S13 energy model).

Single source of truth consumed by both the env and the oracle DP, so
regret stays exact under either cost model. With no power model, the
plant's price-based cost passes through unchanged; with one, cost is
watts x electricity price, where the serving node draws load-proportional
power and a co-billed transitional node idles.
"""

from __future__ import annotations

import numpy as np

from fonpr.sim.config import EconConfig, PlantConfig


def series_cost_and_energy(
    econ: EconConfig,
    plant: PlantConfig,
    tick_minutes: float,
    served: np.ndarray,
    large_on: np.ndarray,
    small_on: np.ndarray,
    serving_large: np.ndarray,
    price_model_cost: float,
) -> tuple[float, float]:
    """Return (infra_cost_usd, energy_wh) for one step's tick series.

    ``price_model_cost`` is the plant's precomputed price-table cost; it is
    returned untouched when no power model is configured (energy 0.0).
    """
    if econ.power is None:
        return price_model_cost, 0.0

    power = econ.power
    tick_hours = tick_minutes / 60.0
    serving_type = np.where(
        serving_large > 0.5, plant.large_instance_type, plant.small_instance_type
    )
    nominal = np.where(
        serving_large > 0.5,
        plant.large_capacity_bytes_per_sec,
        plant.small_capacity_bytes_per_sec,
    )
    utilization = np.clip(served / nominal, 0.0, 1.0)

    energy_wh = 0.0
    for i in range(len(served)):
        watts = power.watts(str(serving_type[i]), float(utilization[i]))
        # A second billed node (transition in flight) idles while it warms
        # or drains.
        if large_on[i] > 0.5 and small_on[i] > 0.5:
            other = (
                plant.small_instance_type
                if serving_large[i] > 0.5
                else plant.large_instance_type
            )
            watts += power.idle_watts[other]
        energy_wh += watts * tick_hours

    cost = energy_wh / 1000.0 * power.electricity_usd_per_kwh
    return cost, energy_wh
