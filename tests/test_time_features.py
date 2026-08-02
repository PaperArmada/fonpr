"""
Tests for the ADR-0002 time-of-day observation variant.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from fonpr.policies import make_baselines
from fonpr.policies.base import COL_COS_TOD, COL_SIN_TOD
from fonpr.sim import ACTION_NOOP, FONPRSimEnv, SimConfig
from fonpr.sim.config import TimeConfig


def enriched_config(**time_kwargs) -> SimConfig:
    time_kwargs.setdefault("episode_days", 0.5)
    return SimConfig(time=TimeConfig(include_time_features=True, **time_kwargs))


def test_default_shape_unchanged():
    env = FONPRSimEnv(SimConfig())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (15, 3)


def test_enriched_shape_and_check_env():
    env = FONPRSimEnv(enriched_config())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (15, 5)
    check_env(FONPRSimEnv(enriched_config()), skip_render_check=True)


def test_time_columns_encode_time_of_day():
    env = FONPRSimEnv(enriched_config())
    obs, _ = env.reset(seed=0)
    # Warm-up covers ticks 0..14 (minutes 0..14 of the sim day).
    minutes = np.arange(15)
    angle = 2 * np.pi * minutes / 1440.0
    np.testing.assert_allclose(obs[:, COL_SIN_TOD], np.sin(angle), rtol=1e-5)
    np.testing.assert_allclose(obs[:, COL_COS_TOD], np.cos(angle), rtol=1e-5)
    # After one step the window covers minutes 15..29.
    obs, _, _, _, _ = env.step(ACTION_NOOP)
    minutes = np.arange(15, 30)
    angle = 2 * np.pi * minutes / 1440.0
    np.testing.assert_allclose(obs[:, COL_SIN_TOD], np.sin(angle), rtol=1e-5)


def test_sin_cos_continuous_at_midnight():
    env = FONPRSimEnv(enriched_config(episode_days=1.5))
    env.reset(seed=0)
    values = []
    for _ in range(env.config.time.episode_steps):
        obs, _, _, truncated, _ = env.step(ACTION_NOOP)
        values.append((obs[-1, COL_SIN_TOD], obs[-1, COL_COS_TOD]))
        if truncated:
            break
    arr = np.asarray(values)
    # Unit circle throughout: no discontinuity artifacts.
    np.testing.assert_allclose(np.hypot(arr[:, 0], arr[:, 1]), 1.0, rtol=1e-5)


def test_rewards_identical_across_variants():
    """Time features change only the observation, never the dynamics."""
    actions = [0, 1, 0, 2, 0] * 4
    plain = FONPRSimEnv(SimConfig(time=TimeConfig(episode_days=0.5)))
    rich = FONPRSimEnv(enriched_config())
    plain.reset(seed=7)
    rich.reset(seed=7)
    for action in actions:
        _, r1, _, _, _ = plain.step(action)
        _, r2, _, _, _ = rich.step(action)
        assert r1 == pytest.approx(r2)


def test_baselines_ignore_extra_columns():
    cfg = enriched_config()
    env = FONPRSimEnv(cfg)
    for policy in make_baselines(cfg):
        policy.reset()
        obs, _ = env.reset(seed=3)
        for _ in range(4):
            obs, _, _, _, _ = env.step(policy.act(obs))
