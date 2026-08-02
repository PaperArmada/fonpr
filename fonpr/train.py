"""
DQN training in the simulator (S5). Requires the [rl] extra.

Checkpoints carry the SimConfig they were trained against, so a policy can
never be silently evaluated on a different plant than it learned.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

from fonpr.policies.base import Policy
from fonpr.sim import FONPRSimEnv, SimConfig

logger = logging.getLogger(__name__)

_SB3_HINT = "stable-baselines3 is not installed; install the [rl] extra (pip install 'fonpr[rl]')"


def _require_sb3():
    try:
        from stable_baselines3 import DQN
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_SB3_HINT) from exc
    return DQN


class FlattenObsWrapper:
    """Gymnasium wrapper flattening (samples, 3) to 1-D for the MLP policy."""

    def __new__(cls, env):
        import gymnasium as gym

        return gym.wrappers.FlattenObservation(env)


def train_dqn(
    config: SimConfig,
    steps: int,
    seed: int,
    out_dir: Path,
    checkpoint_every: int = 50_000,
) -> Path:
    """Train DQN per S5 and return the final checkpoint path."""
    dqn_cls = _require_sb3()
    from stable_baselines3.common.monitor import Monitor

    out_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(FlattenObsWrapper(FONPRSimEnv(config)))
    env.reset(seed=seed)

    model = dqn_cls(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [64, 64]},
        seed=seed,
        verbose=0,
    )
    remaining, trained = steps, 0
    while remaining > 0:
        chunk = min(checkpoint_every, remaining)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        trained += chunk
        remaining -= chunk
        model.save(out_dir / f"checkpoint_{trained:07d}")
        logger.info("trained %d/%d steps", trained, steps)

    final = out_dir / "final"
    model.save(final)
    config.to_yaml(out_dir / "sim_config.yaml")
    with open(out_dir / "train_meta.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump({"steps": steps, "seed": seed, "algorithm": "DQN"}, fh)
    return Path(str(final) + ".zip")


class DQNPolicy(Policy):
    """Deploys a trained checkpoint behind the Policy seam (deterministic)."""

    name = "dqn"

    def __init__(self, checkpoint: str | Path):
        dqn_cls = _require_sb3()
        self._model = dqn_cls.load(checkpoint)

    def act(self, obs: np.ndarray) -> int:
        action, _ = self._model.predict(obs.reshape(-1), deterministic=True)
        return int(action)


def run_train_cli(config_path: str | None, out_root: str, steps: int, seed: int) -> int:
    """Entry point behind ``fonpr train``."""
    config = SimConfig.from_yaml(config_path) if config_path else SimConfig()
    out_dir = Path(out_root) / f"dqn-seed{seed}"
    final = train_dqn(config, steps=steps, seed=seed, out_dir=out_dir)
    logger.info("final checkpoint: %s", final)
    return 0
