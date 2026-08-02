"""
Smoke test for DQN training (S5/S8): a short run must produce a loadable
checkpoint that acts through the Policy seam. Skipped without the [rl] extra.
"""

import pytest

sb3 = pytest.importorskip("stable_baselines3")

import pytest  # noqa: E402

from fonpr.sim import FONPRSimEnv, SimConfig  # noqa: E402
from fonpr.sim.config import TimeConfig  # noqa: E402
from fonpr.train import AgentPolicy, train_agent  # noqa: E402


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_smoke_train_and_deploy(tmp_path, algorithm):
    config = SimConfig(time=TimeConfig(episode_days=1.0))
    final = train_agent(
        config, steps=500, seed=0, out_dir=tmp_path, algorithm=algorithm, checkpoint_every=500
    )
    assert final.exists()
    assert (tmp_path / "sim_config.yaml").exists()

    # Algorithm read back from train_meta.yaml — no explicit class needed.
    policy = AgentPolicy(final)
    env = FONPRSimEnv(config)
    obs, _ = env.reset(seed=1)
    for _ in range(3):
        action = policy.act(obs)
        assert action in (0, 1, 2)
        obs, _, _, _, _ = env.step(action)


def test_unknown_algorithm_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown algorithm"):
        train_agent(SimConfig(), steps=10, seed=0, out_dir=tmp_path, algorithm="sac")
