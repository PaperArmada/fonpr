"""
Actuator backends (S6): dry-run (default everywhere), direct Helm (local
dev), GitHub+Flux (production GitOps path, wrapping the legacy handler).
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

from fonpr.actuators.base import ActuationResult, Actuator

logger = logging.getLogger(__name__)


class DryRunActuator(Actuator):
    """Shadow mode: log the would-be change, apply nothing.

    The default in every config template (CLAUDE.md invariant).
    """

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def apply(self, requested_actions: dict[str, Any]) -> ActuationResult:
        self.history.append(requested_actions)
        logger.info("[dry-run] would apply: %s", requested_actions)
        return ActuationResult(success=True, applied=requested_actions, reference="dry-run")


class HelmActuator(Actuator):
    """Direct ``helm upgrade --reuse-values --set k=v`` against the current
    kube-context. For local development clusters only."""

    def __init__(self, release: str, chart: str, namespace: str = "default", timeout_s: int = 120):
        self._release = release
        self._chart = chart
        self._namespace = namespace
        self._timeout_s = timeout_s

    def build_command(self, requested_actions: dict[str, Any]) -> list[str]:
        cmd = [
            "helm",
            "upgrade",
            self._release,
            self._chart,
            "--namespace",
            self._namespace,
            "--reuse-values",
        ]
        for key, value in sorted(_flatten(requested_actions).items()):
            cmd += ["--set", f"{key}={value}"]
        return cmd

    def apply(self, requested_actions: dict[str, Any]) -> ActuationResult:
        cmd = self.build_command(requested_actions)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self._timeout_s, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("helm actuation failed to run: %s", exc)
            return ActuationResult(success=False, error=str(exc))
        if proc.returncode != 0:
            logger.warning("helm actuation failed (rc=%d): %s", proc.returncode, proc.stderr)
            return ActuationResult(success=False, error=proc.stderr.strip())
        return ActuationResult(
            success=True, applied=requested_actions, reference=f"helm:{self._release}"
        )


class GitHubFluxActuator(Actuator):
    """Production GitOps path: commit value updates to the controlling repo;
    Flux reconciles the cluster. Wraps the legacy ActionHandler (requires the
    [live] extra); the commit is the audit record."""

    def __init__(
        self,
        token: str,
        repo_name: str,
        branch_name: str,
        value_file_dir: str,
        value_file_name: str,
    ):
        from action_handler import ActionHandler  # optional [live] dependency path

        self._handler = ActionHandler(
            token, repo_name, branch_name, value_file_dir, value_file_name
        )
        self._target = f"{repo_name}:{branch_name}/{value_file_dir}/{value_file_name}"

    def apply(self, requested_actions: dict[str, Any]) -> ActuationResult:
        try:
            self._handler.set_requested_actions(requested_actions)
            self._handler.fetch_update_push()
        except Exception as exc:  # backend surface is broad: network, auth, API
            logger.warning("github actuation failed: %s", exc)
            return ActuationResult(success=False, error=str(exc))
        return ActuationResult(success=True, applied=requested_actions, reference=self._target)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out
