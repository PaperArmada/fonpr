"""
Actuator seam (S6): requested action -> applied change.

Operational failures never raise out of ``apply``: they return a failed
ActuationResult, which the agent control loop treats as a no-op step with a
logged warning. Crashing the control loop is worse than skipping a step.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ActuationResult:
    success: bool
    applied: dict[str, Any] = field(default_factory=dict)
    reference: str | None = None  # backend handle: commit SHA, helm release, ...
    error: str | None = None


class Actuator(ABC):
    """One ``apply`` per agent decision; idempotent per call."""

    @abstractmethod
    def apply(self, requested_actions: dict[str, Any]) -> ActuationResult:
        """Apply the requested value updates; never raise for operational failures."""


def get_repo_token(env_var: str = "GH_TOKEN", file_env_var: str = "GH_TOKEN_FILE") -> str:
    """Resolve the GitHub token with S6 precedence: env var, token file, AWS.

    AWS Secrets Manager is the last resort and requires the [live] extra.
    """
    token = os.environ.get(env_var)
    if token:
        return token
    token_file = os.environ.get(file_env_var)
    if token_file:
        return Path(token_file).read_text(encoding="utf-8").strip()
    from action_handler import get_token  # optional [live] dependency path

    return get_token()
