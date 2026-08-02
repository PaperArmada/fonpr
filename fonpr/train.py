"""
Agent training in the simulator (S5). Requires the [rl] extra.

Two supported algorithms: DQN (ADR-0001/D2, the ratified default) and PPO
(candidate under evaluation after the 2026-08-02 campaign exposed DQN
training instability; adoption would supersede ADR-0001/D2).

Checkpoints carry the SimConfig they were trained against, so a policy can
never be silently evaluated on a different plant than it learned. The
train_meta.yaml records the algorithm; the eval CLI needs it to load the
checkpoint with the right class.
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

ALGORITHMS = ("dqn", "ppo")


def _algo_class(algorithm: str):
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_SB3_HINT) from exc
    try:
        return {"dqn": DQN, "ppo": PPO}[algorithm]
    except KeyError:
        raise ValueError(f"unknown algorithm {algorithm!r}; expected one of {ALGORITHMS}") from None


class FlattenObsWrapper:
    """Gymnasium wrapper flattening (samples, cols) to 1-D for the MLP policy."""

    def __new__(cls, env):
        import gymnasium as gym

        return gym.wrappers.FlattenObservation(env)


def train_agent(
    config: SimConfig,
    steps: int,
    seed: int,
    out_dir: Path,
    algorithm: str = "dqn",
    checkpoint_every: int = 50_000,
) -> Path:
    """Train per S5 and return the final checkpoint path."""
    algo_cls = _algo_class(algorithm)
    from stable_baselines3.common.monitor import Monitor

    out_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(FlattenObsWrapper(FONPRSimEnv(config)))
    env.reset(seed=seed)

    model = algo_cls(
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
        yaml.safe_dump({"steps": steps, "seed": seed, "algorithm": algorithm}, fh)
    return Path(str(final) + ".zip")


class AgentPolicy(Policy):
    """Deploys a trained checkpoint behind the Policy seam (deterministic).

    ``algorithm`` selects the loader class; when omitted, it is read from
    the train_meta.yaml next to the checkpoint, falling back to 'dqn'.
    """

    name = "agent"

    def __init__(self, checkpoint: str | Path, algorithm: str | None = None):
        checkpoint = Path(checkpoint)
        if algorithm is None:
            meta_path = checkpoint.parent / "train_meta.yaml"
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as fh:
                    algorithm = (yaml.safe_load(fh) or {}).get("algorithm", "dqn")
            else:
                algorithm = "dqn"
        self._model = _algo_class(algorithm).load(checkpoint)

    def act(self, obs: np.ndarray) -> int:
        action, _ = self._model.predict(obs.reshape(-1), deterministic=True)
        return int(action)


class DQNPolicy(AgentPolicy):
    """Backwards-compatible alias for DQN checkpoints."""

    name = "dqn"

    def __init__(self, checkpoint: str | Path):
        super().__init__(checkpoint, algorithm=None)


def run_train_cli(
    config_path: str | None, out_root: str, steps: int, seed: int, algorithm: str = "dqn"
) -> int:
    """Entry point behind ``fonpr train``."""
    config = SimConfig.from_yaml(config_path) if config_path else SimConfig()
    out_dir = Path(out_root) / f"{algorithm}-seed{seed}"
    final = train_agent(config, steps=steps, seed=seed, out_dir=out_dir, algorithm=algorithm)
    logger.info("final checkpoint: %s", final)
    return 0
