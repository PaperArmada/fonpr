"""
Live control loop (S12): Advisor -> Policy -> Actuator on a fixed cadence.

The deployment counterpart of one sim step. Observation shape and action
semantics match S1.1 exactly, so any Policy — baseline or trained
checkpoint — runs here unchanged. DryRunActuator is the default; a live
actuator runs only when the config names one explicitly.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fonpr.actuators import Actuator, DryRunActuator, GitHubFluxActuator, HelmActuator
from fonpr.actuators.base import get_repo_token
from fonpr.policies import (
    ForecastPolicy,
    NoopPolicy,
    Policy,
    ReactivePolicy,
    ThresholdPolicy,
)
from fonpr.sim import ACTION_LARGE, ACTION_NOOP, SimConfig

logger = logging.getLogger(__name__)

POLICY_NAMES = ("noop", "threshold", "reactive", "forecast", "checkpoint")
ACTUATOR_KINDS = ("dry-run", "helm", "github")


@dataclass(frozen=True)
class LoopConfig:
    """Config for `fonpr run-agent`. YAML-loadable; strict keys.

    ``sim`` supplies the shared vocabulary (window, instance types,
    capacities) so baselines threshold on the same numbers they were
    benchmarked with.
    """

    prom_endpoint: str = "localhost:9090"
    policy: str = "threshold"
    checkpoint_path: str | None = None
    actuator: str = "dry-run"  # doctrine: dry-run default, always
    interval_minutes: float = 15.0
    # helm backend
    helm_release: str = "respons"
    helm_chart: str = "charts/respons"
    helm_namespace: str = "default"
    # github backend
    github_repo: str = ""
    github_branch: str = ""
    github_value_file_dir: str = ""
    github_value_file_name: str = ""
    sim: SimConfig = field(default_factory=SimConfig)

    def __post_init__(self) -> None:
        if self.policy not in POLICY_NAMES:
            raise ValueError(f"policy must be one of {POLICY_NAMES}")
        if self.policy == "checkpoint" and not self.checkpoint_path:
            raise ValueError("policy 'checkpoint' requires checkpoint_path")
        if self.actuator not in ACTUATOR_KINDS:
            raise ValueError(f"actuator must be one of {ACTUATOR_KINDS}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> LoopConfig:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise KeyError(f"loop config: unknown key(s) {sorted(unknown)}")
        if "sim" in raw:
            raw["sim"] = SimConfig.from_dict(raw["sim"])
        return cls(**raw)


def build_policy(cfg: LoopConfig) -> Policy:
    if cfg.policy == "noop":
        return NoopPolicy()
    if cfg.policy == "threshold":
        return ThresholdPolicy(cfg.sim)
    if cfg.policy == "reactive":
        return ReactivePolicy(cfg.sim)
    if cfg.policy == "forecast":
        return ForecastPolicy(cfg.sim)
    from fonpr.train import AgentPolicy  # [rl] extra

    return AgentPolicy(cfg.checkpoint_path)


def build_actuator(cfg: LoopConfig) -> Actuator:
    if cfg.actuator == "dry-run":
        return DryRunActuator()
    if cfg.actuator == "helm":
        return HelmActuator(cfg.helm_release, cfg.helm_chart, cfg.helm_namespace)
    return GitHubFluxActuator(
        token=get_repo_token(),
        repo_name=cfg.github_repo,
        branch_name=cfg.github_branch,
        value_file_dir=cfg.github_value_file_dir,
        value_file_name=cfg.github_value_file_name,
    )


def action_to_request(action: int, cfg: LoopConfig) -> dict[str, Any] | None:
    """Map a Policy action to an actuation request; None means no actuation.

    The request pins the UPF to the chosen node group via its instance-type
    node selector — the live counterpart of the sim's plant transition.
    """
    if action == ACTION_NOOP:
        return None
    instance_type = (
        cfg.sim.plant.large_instance_type
        if action == ACTION_LARGE
        else cfg.sim.plant.small_instance_type
    )
    return {
        "target_pod": "upf",
        "nodeSelector": {"node.kubernetes.io/instance-type": instance_type},
    }


def run_loop(
    cfg: LoopConfig,
    max_iterations: int | None = None,
    advisor=None,
    sleep=time.sleep,
) -> list[dict[str, Any]]:
    """Run the control loop; returns the per-iteration log (for tests/audit).

    ``advisor`` and ``sleep`` are injectable for testing; the default
    advisor is the live ThroughputAdvisor against cfg.prom_endpoint.
    """
    if advisor is None:
        from fonpr.advisors.throughput_advisor import ThroughputAdvisor

        advisor = ThroughputAdvisor(
            cfg.prom_endpoint,
            cfg.sim.plant.large_instance_type,
            cfg.sim.plant.small_instance_type,
        )
    policy = build_policy(cfg)
    actuator = build_actuator(cfg)
    policy.reset()
    logger.info(
        "control loop: policy=%s actuator=%s interval=%.1fmin",
        cfg.policy,
        cfg.actuator,
        cfg.interval_minutes,
    )

    history: list[dict[str, Any]] = []
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        record: dict[str, Any] = {"iteration": iteration}
        try:
            obs = advisor.observe(
                cfg.sim.time.window_minutes, cfg.sim.time.sample_rate_per_minute
            )
            action = policy.act(obs)
            request = action_to_request(action, cfg)
            record.update(action=action, request=request)
            if request is not None:
                result = actuator.apply(request)
                record.update(applied=result.success, reference=result.reference)
                if not result.success:
                    logger.warning("actuation failed (treated as no-op): %s", result.error)
            else:
                record.update(applied=False, reference=None)
        except Exception as exc:
            # The control loop is the one sanctioned broad catch (CLAUDE.md):
            # a failed observation skips the iteration, never kills the loop.
            logger.warning("iteration %d failed, skipping: %s", iteration, exc)
            record.update(error=str(exc))
        history.append(record)
        logger.info("iteration %d: %s", iteration, record)
        if max_iterations is None or iteration < max_iterations:
            sleep(cfg.interval_minutes * 60.0)
    return history


def run_agent_cli(config_path: str | None, once: bool = False) -> int:
    """Entry point behind ``fonpr run-agent``."""
    cfg = LoopConfig.from_yaml(config_path) if config_path else LoopConfig()
    run_loop(cfg, max_iterations=1 if once else None)
    return 0
