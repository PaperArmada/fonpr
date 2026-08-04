"""
Evaluation harness (S4): run every policy over every scenario x seed,
compute regret against the hindsight oracle, and write reproducible
result bundles. This is the only source of performance claims.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fonpr.policies import OraclePolicy, Policy, make_baselines, plan_oracle_actions
from fonpr.sim import FONPRSimEnv, SimConfig
from fonpr.sim.config import (
    EconConfig,
    PlantConfig,
    PoolConfig,
    PowerConfig,
    TimeConfig,
    TrafficConfig,
)

logger = logging.getLogger(__name__)

SCENARIOS = ("steady", "diurnal", "diurnal_bursty", "drift")
HEADLINE_SCENARIO = "diurnal_bursty"


@dataclass(frozen=True)
class PerturbConfig:
    """One robustness cell (S4.5, ADR-0005 rung 3): how the evaluation
    world differs from the nominal twin the policies believe in."""

    # Multiplies the world plant's capacities (both tiers / node capacity).
    capacity_scale: float = 1.0
    # Multiplies the world plant's transition lag.
    lag_scale: float = 1.0
    # Multiplicative iid gaussian on the observed served-throughput column
    # only, between world and policy. Fleet-state columns stay exact: an
    # operator knows its own fleet.
    obs_noise_sigma_frac: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity_scale <= 0 or self.lag_scale <= 0:
            raise ValueError("capacity_scale and lag_scale must be positive")
        if self.obs_noise_sigma_frac < 0:
            raise ValueError("obs_noise_sigma_frac must be non-negative")


@dataclass(frozen=True)
class EvalConfig:
    """Harness parameters (S4.2). Scenario definitions live in scenario_config."""

    scenarios: tuple[str, ...] = SCENARIOS
    n_seeds: int = 20
    seed_offset: int = 10_000  # eval seeds disjoint from training seeds by construction
    episode_days: float = 7.0
    # ADR-0002: evaluate on the enriched observation. Traffic, plant, and
    # seeds are unaffected, so costs stay comparable across variants.
    include_time_features: bool = False
    # S13: derive infra cost from the power model (PowerConfig defaults,
    # matching configs/sim-energy.yaml) instead of the cloud price table.
    energy: bool = False
    # S14/ADR-0004: run the pool plant (PoolConfig defaults, matching
    # configs/sim-pool.yaml). Pool costs compare only within pool bundles.
    pool: bool = False
    # S4.5 (ADR-0005 rung 3): label -> robustness cell. None runs the S4.2
    # protocol unchanged. Labels are filename-safe ([A-Za-z0-9_-]).
    perturbations: dict[str, PerturbConfig] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise KeyError(f"eval config: unknown key(s) {sorted(unknown)}")
        if "scenarios" in raw:
            raw["scenarios"] = tuple(raw["scenarios"])
        if raw.get("perturbations") is not None:
            cells = {}
            for label, cell in raw["perturbations"].items():
                if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
                    raise ValueError(f"perturbation label {label!r} is not filename-safe")
                cell_known = {f.name for f in dataclasses.fields(PerturbConfig)}
                cell_unknown = set(cell or {}) - cell_known
                if cell_unknown:
                    raise KeyError(
                        f"perturbation {label!r}: unknown key(s) {sorted(cell_unknown)}"
                    )
                cells[label] = PerturbConfig(**(cell or {}))
            raw["perturbations"] = cells
        return cls(**raw)


def scenario_config(
    name: str,
    episode_days: float,
    include_time_features: bool = False,
    energy: bool = False,
    pool: bool = False,
) -> SimConfig:
    """Named traffic scenarios (S4.2) over the default or pool plant (S14),
    under either the price-table econ (default) or the S13 power econ."""
    time = TimeConfig(episode_days=episode_days, include_time_features=include_time_features)
    if name == "steady":
        traffic = TrafficConfig(diurnal_amplitude=0.0, burst_rate_per_day=0.0)
    elif name == "diurnal":
        traffic = TrafficConfig(burst_rate_per_day=0.0)
    elif name == "diurnal_bursty":
        traffic = TrafficConfig()
    elif name == "drift":
        traffic = TrafficConfig(drift_frac_per_day=0.05)
    else:
        raise ValueError(f"unknown scenario {name!r}")
    econ = EconConfig(power=PowerConfig()) if energy else EconConfig()
    plant = PlantConfig(pool=PoolConfig()) if pool else PlantConfig()
    return SimConfig(time=time, traffic=traffic, plant=plant, econ=econ)


def perturbed_world(belief: SimConfig, cell: PerturbConfig) -> SimConfig:
    """The evaluation world for one robustness cell (S4.5).

    Policies keep their beliefs (thresholds, forecasts, MPC's internal cost
    model, a checkpoint's weights) from the nominal config; only the world
    the env and the oracle run changes. Lag stays integer-typed, rounded.
    """
    plant = belief.plant
    if plant.pool is not None:
        pool = dataclasses.replace(
            plant.pool,
            node_capacity_bytes_per_sec=(
                plant.pool.node_capacity_bytes_per_sec * cell.capacity_scale
            ),
            transition_lag_minutes=max(
                1, round(plant.pool.transition_lag_minutes * cell.lag_scale)
            ),
        )
        plant = dataclasses.replace(plant, pool=pool)
    else:
        plant = dataclasses.replace(
            plant,
            small_capacity_bytes_per_sec=(
                plant.small_capacity_bytes_per_sec * cell.capacity_scale
            ),
            large_capacity_bytes_per_sec=(
                plant.large_capacity_bytes_per_sec * cell.capacity_scale
            ),
            transition_lag_minutes=max(
                1, round(plant.transition_lag_minutes * cell.lag_scale)
            ),
        )
    return dataclasses.replace(belief, plant=plant)


class _NoisyObsPolicy(Policy):
    """Sensor error between world and controller (S4.5): multiplicative iid
    gaussian on the served-throughput column, deterministic per (eval seed,
    policy name) via a stable digest — never the process hash seed."""

    def __init__(self, inner: Policy, sigma_frac: float, seed: int):
        self._inner = inner
        self.name = inner.name
        self._sigma = sigma_frac
        self._seed_key = [seed, zlib.crc32(inner.name.encode("utf-8"))]
        self.reset()

    def reset(self) -> None:
        self._inner.reset()
        self._rng = np.random.default_rng(self._seed_key)

    def act(self, obs: np.ndarray) -> int:
        noisy = obs.copy()
        factors = 1.0 + self._rng.normal(0.0, self._sigma, size=len(noisy))
        noisy[:, 0] = np.maximum(noisy[:, 0] * factors, 0.0).astype(obs.dtype)
        return self._inner.act(noisy)


@dataclass
class EpisodeRecord:
    scenario: str
    policy: str
    seed: int
    metrics: dict[str, float]
    # Per-tick series for the timeline plot (kept only for seed index 0).
    series: dict[str, np.ndarray] | None = None
    perturb: str = ""  # robustness-cell label; empty under the S4.2 protocol


def run_episode(
    env: FONPRSimEnv, policy: Policy, seed: int, keep_series: bool = False
) -> tuple[dict[str, float], list[np.ndarray], dict[str, np.ndarray] | None]:
    """Roll one full episode; return (metrics, offered_steps, series)."""
    policy.reset()
    obs, _ = env.reset(seed=seed)
    infra = penalty = violation_minutes = energy_wh = 0.0
    churn = 0
    offered_sum = served_sum = 0.0
    offered_steps: list[np.ndarray] = []
    series: dict[str, list[np.ndarray]] = {"offered": [], "served": [], "capacity": []}

    truncated = False
    while not truncated:
        action = policy.act(obs)
        obs, _, _, truncated, info = env.step(action)
        infra += info["step_cost_usd"]
        penalty += info["step_penalty_usd"]
        energy_wh += info["step_energy_wh"]
        violation_minutes += info["slo_violation"]
        churn += int(info["action_applied"])
        offered_sum += float(np.sum(info["offered_series"]))
        served_sum += float(np.sum(info["served_series"]))
        offered_steps.append(info["offered_series"])
        if keep_series:
            series["offered"].append(info["offered_series"])
            series["served"].append(info["served_series"])
            series["capacity"].append(info["capacity_series"])

    episode_minutes = env.config.time.episode_steps * env.config.time.step_minutes
    metrics = {
        "infra_cost_usd": infra,
        "penalty_usd": penalty,
        "total_cost_usd": infra + penalty,
        "violation_minutes": violation_minutes,
        "violation_fraction": violation_minutes / episode_minutes,
        "action_churn": float(churn),
        "served_offered_ratio": served_sum / offered_sum if offered_sum else 1.0,
        "energy_kwh": energy_wh / 1000.0,  # 0.0 under the price model (S13)
    }
    kept = (
        {name: np.concatenate(chunks) for name, chunks in series.items()}
        if keep_series
        else None
    )
    return metrics, offered_steps, kept


def evaluate(
    eval_cfg: EvalConfig, extra_policies: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Run the full grid. Returns (tidy results frame, timeline series).

    ``extra_policies`` maps name -> factory(SimConfig) -> Policy, letting the
    training pipeline add learned agents to the standard suite.
    """
    records: list[EpisodeRecord] = []
    timelines: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    # No perturbations: one unlabeled cell, the S4.2 protocol unchanged.
    cells: dict[str, PerturbConfig | None] = (
        dict(eval_cfg.perturbations) if eval_cfg.perturbations else {"": None}
    )

    for scenario in eval_cfg.scenarios:
        belief_cfg = scenario_config(
            scenario,
            eval_cfg.episode_days,
            eval_cfg.include_time_features,
            eval_cfg.energy,
            eval_cfg.pool,
        )
        for cell_label, cell in cells.items():
            world_cfg = (
                belief_cfg if cell is None else perturbed_world(belief_cfg, cell)
            )
            env = FONPRSimEnv(world_cfg)
            display = scenario if not cell_label else f"{scenario}__{cell_label}"
            timelines[display] = {}

            for i in range(eval_cfg.n_seeds):
                seed = eval_cfg.seed_offset + i
                keep = i == 0
                # Policies believe the nominal twin (S4.5); only the world
                # they are dropped into changes.
                policies: list[Policy] = make_baselines(belief_cfg)
                for name, factory in (extra_policies or {}).items():
                    p = factory(belief_cfg)
                    p.name = name
                    policies.append(p)
                if cell is not None and cell.obs_noise_sigma_frac > 0:
                    policies = [
                        _NoisyObsPolicy(p, cell.obs_noise_sigma_frac, seed)
                        for p in policies
                    ]

                offered_steps: list[np.ndarray] | None = None
                oracle_cost: float | None = None
                seed_records: list[EpisodeRecord] = []

                for policy in policies:
                    metrics, offered, series = run_episode(
                        env, policy, seed, keep_series=keep
                    )
                    offered_steps = offered_steps or offered
                    seed_records.append(
                        EpisodeRecord(
                            scenario, policy.name, seed, metrics, series, cell_label
                        )
                    )

                # Oracle on the identical offered-load trace (traffic is
                # action-independent, so any policy's trace is THE trace).
                # It plans on the WORLD: regret is against what was truly
                # achievable, and it sees no observation noise (S4.5).
                actions, planned = plan_oracle_actions(offered_steps, world_cfg)
                metrics, _, series = run_episode(
                    env, OraclePolicy(actions), seed, keep_series=keep
                )
                oracle_cost = metrics["total_cost_usd"]
                if abs(planned - oracle_cost) > 1e-6:
                    raise AssertionError(
                        f"oracle DP/plant divergence: planned {planned}, realized {oracle_cost}"
                    )
                seed_records.append(
                    EpisodeRecord(scenario, "oracle", seed, metrics, series, cell_label)
                )

                for rec in seed_records:
                    rec.metrics["regret_usd"] = rec.metrics["total_cost_usd"] - oracle_cost
                    records.append(rec)
                    if keep and rec.series is not None:
                        timelines[display][rec.policy] = rec.series
                logger.info(
                    "scenario=%s%s seed=%d done",
                    scenario,
                    f" cell={cell_label}" if cell_label else "",
                    seed,
                )

    rows = [
        {
            "scenario": r.scenario,
            "perturb": r.perturb,
            "policy": r.policy,
            "seed": r.seed,
            **r.metrics,
        }
        for r in records
    ]
    return pd.DataFrame(rows), timelines


def git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_eval_cli(
    config_path: str | None,
    out_root: str,
    quick: bool = False,
    dqn_checkpoint: list[str] | None = None,
    time_features: bool = False,
    energy: bool = False,
    pool: bool = False,
) -> int:
    """Entry point behind ``fonpr eval`` (S4.3)."""
    from fonpr.eval.report import write_report

    if config_path:
        eval_cfg = EvalConfig.from_yaml(config_path)
    elif quick:
        eval_cfg = EvalConfig(
            scenarios=("steady", HEADLINE_SCENARIO),
            n_seeds=3,
            episode_days=2.0,
            include_time_features=time_features,
            energy=energy,
            pool=pool,
        )
    else:
        eval_cfg = EvalConfig(include_time_features=time_features, energy=energy, pool=pool)

    extra_policies = None
    if dqn_checkpoint:
        # AgentPolicy reads the algorithm from train_meta.yaml next to the
        # checkpoint, so DQN and PPO checkpoints mix freely here.
        from fonpr.train import AgentPolicy

        extra_policies = {}
        for spec in dqn_checkpoint:
            label, _, path = spec.rpartition("=")
            label = label or "dqn"
            if label in extra_policies:
                raise ValueError(f"duplicate policy label {label!r}")
            extra_policies[label] = (
                lambda _cfg, p=path: AgentPolicy(p)  # bind path per iteration
            )

    results, timelines = evaluate(eval_cfg, extra_policies=extra_policies)
    out_dir = write_report(results, timelines, eval_cfg, Path(out_root))
    logger.info("results written to %s", out_dir)
    return 0
