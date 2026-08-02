"""
Tests for the simulation environment (S1): determinism, Gymnasium
compliance, plant economics, and traffic generation.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from fonpr.sim import (
    ACTION_LARGE,
    ACTION_NOOP,
    ACTION_SMALL,
    EconConfig,
    FONPRSimEnv,
    PlantConfig,
    PlantModel,
    SimConfig,
    TimeConfig,
    TrafficConfig,
    TrafficModel,
)


def short_config(**traffic_overrides) -> SimConfig:
    return SimConfig(
        time=TimeConfig(episode_days=0.5),
        traffic=TrafficConfig(**traffic_overrides),
    )


def rollout(env, seed, actions):
    obs, _ = env.reset(seed=seed)
    trajectory = [obs.copy()]
    rewards = []
    for a in actions:
        obs, reward, _, truncated, _ = env.step(a)
        trajectory.append(obs.copy())
        rewards.append(reward)
        if truncated:
            break
    return trajectory, rewards


class TestDeterminism:
    def test_same_seed_same_trajectory(self):
        actions = [ACTION_NOOP, ACTION_LARGE, ACTION_NOOP, ACTION_SMALL] * 5
        t1, r1 = rollout(FONPRSimEnv(short_config()), 123, actions)
        t2, r2 = rollout(FONPRSimEnv(short_config()), 123, actions)
        assert r1 == r2
        for a, b in zip(t1, t2, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_different_seed_differs(self):
        # Rewards can coincide (violation minutes are quantized) and saturated
        # load clips to capacity, so compare unsaturated observed throughput.
        actions = [ACTION_NOOP] * 5
        cfg = dict(base_load_bytes_per_sec=5e6)  # below small capacity
        t1, _ = rollout(FONPRSimEnv(short_config(**cfg)), 1, actions)
        t2, _ = rollout(FONPRSimEnv(short_config(**cfg)), 2, actions)
        assert any(not np.array_equal(a, b) for a, b in zip(t1, t2, strict=True))


class TestGymCompliance:
    def test_check_env(self):
        check_env(FONPRSimEnv(short_config()), skip_render_check=True)

    def test_observation_shape_and_dtype(self):
        env = FONPRSimEnv()
        obs, info = env.reset(seed=0)
        assert obs.shape == (env.config.time.window_ticks, 3)
        assert obs.dtype == np.float32
        assert set(info) >= {
            "offered_load",
            "served_load",
            "slo_violation",
            "instance_type",
            "in_transition",
            "step_cost_usd",
            "step_penalty_usd",
        }

    def test_episode_truncates_never_terminates(self):
        cfg = short_config()
        env = FONPRSimEnv(cfg)
        env.reset(seed=0)
        steps_taken = 0
        for _ in range(cfg.time.episode_steps):
            _, _, terminated, truncated, _ = env.step(ACTION_NOOP)
            assert not terminated
            steps_taken += 1
        assert truncated
        assert steps_taken == cfg.time.episode_steps

    def test_invalid_action_raises(self):
        env = FONPRSimEnv(short_config())
        env.reset(seed=0)
        with pytest.raises(ValueError):
            env.step(7)


class TestPlantEconomics:
    def make_plant(self, lag_minutes=5):
        return PlantModel(
            PlantConfig(transition_lag_minutes=lag_minutes), EconConfig(), TimeConfig()
        )

    def test_saturation_drops_excess_load(self):
        plant = self.make_plant()
        offered = np.full(15, 1e9)  # far above small capacity
        result = plant.advance(offered)
        np.testing.assert_allclose(result.served, PlantConfig().small_capacity_bytes_per_sec)

    def test_steady_state_billing_small(self):
        plant = self.make_plant()
        result = plant.advance(np.zeros(60))  # one hour on t3.medium
        assert result.infra_cost_usd == pytest.approx(0.0416)
        assert result.small_on.all() and not result.large_on.any()

    def test_transition_double_bills_and_degrades(self):
        plant = self.make_plant(lag_minutes=5)
        assert plant.request_transition("large")
        result = plant.advance(np.full(15, 1e9))
        # First 5 minutes: both billed, small capacity. Remaining 10: large only.
        expected = (0.20 + 0.0416) * 5 / 60 + 0.20 * 10 / 60
        assert result.infra_cost_usd == pytest.approx(expected)
        np.testing.assert_allclose(
            result.served[:5], PlantConfig().small_capacity_bytes_per_sec
        )
        np.testing.assert_allclose(
            result.served[5:], PlantConfig().large_capacity_bytes_per_sec
        )
        assert result.large_on[:5].all() and result.small_on[:5].all()
        assert result.large_on[5:].all() and not result.small_on[5:].any()
        assert plant.active == "large" and not plant.in_transition

    def test_requests_ignored_mid_transition_and_when_redundant(self):
        plant = self.make_plant()
        assert not plant.request_transition("small")  # already small
        assert plant.request_transition("large")
        assert not plant.request_transition("small")  # in flight: ignored
        plant.advance(np.zeros(5))
        assert plant.active == "large"

    def test_zero_lag_transitions_instantly(self):
        plant = self.make_plant(lag_minutes=0)
        assert plant.request_transition("large")
        assert plant.active == "large" and not plant.in_transition


class TestReward:
    def test_reward_is_negative_cost_plus_penalty(self):
        # Deterministic traffic (no noise/bursts) far above small capacity:
        # every minute violates while on small.
        cfg = SimConfig(
            time=TimeConfig(episode_days=0.5),
            traffic=TrafficConfig(
                base_load_bytes_per_sec=5e7,
                diurnal_amplitude=0.0,
                noise_sigma_frac=0.0,
                burst_rate_per_day=0.0,
            ),
        )
        env = FONPRSimEnv(cfg)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(ACTION_NOOP)
        expected_cost = 0.0416 * 15 / 60
        expected_penalty = 15 * cfg.econ.penalty_per_violation_minute_usd
        assert info["step_cost_usd"] == pytest.approx(expected_cost)
        assert info["step_penalty_usd"] == pytest.approx(expected_penalty)
        assert reward == pytest.approx(-(expected_cost + expected_penalty))
        assert info["slo_violation"] == pytest.approx(15.0)

    def test_no_penalty_when_capacity_sufficient(self):
        cfg = SimConfig(
            time=TimeConfig(episode_days=0.5),
            traffic=TrafficConfig(
                base_load_bytes_per_sec=1e6,
                diurnal_amplitude=0.0,
                noise_sigma_frac=0.0,
                burst_rate_per_day=0.0,
            ),
        )
        env = FONPRSimEnv(cfg)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(ACTION_NOOP)
        assert info["step_penalty_usd"] == 0.0
        assert reward == pytest.approx(-info["step_cost_usd"])


class TestTraffic:
    def test_reproducible_with_same_generator_seed(self):
        cfg, tcfg = TrafficConfig(), TimeConfig()
        a = TrafficModel(cfg, tcfg, np.random.default_rng(7)).advance(1000)
        b = TrafficModel(cfg, tcfg, np.random.default_rng(7)).advance(1000)
        np.testing.assert_array_equal(a, b)

    def test_diurnal_shape_without_noise(self):
        cfg = TrafficConfig(
            noise_sigma_frac=0.0, burst_rate_per_day=0.0, diurnal_amplitude=0.5
        )
        loads = TrafficModel(cfg, TimeConfig(), np.random.default_rng(0)).advance(24 * 60)
        assert loads.max() == pytest.approx(1.5 * cfg.base_load_bytes_per_sec, rel=1e-3)
        assert loads.min() == pytest.approx(0.5 * cfg.base_load_bytes_per_sec, rel=1e-3)

    def test_bursts_occur_and_raise_load(self):
        quiet = TrafficConfig(noise_sigma_frac=0.0, burst_rate_per_day=0.0)
        bursty = TrafficConfig(noise_sigma_frac=0.0, burst_rate_per_day=48.0)
        base = TrafficModel(quiet, TimeConfig(), np.random.default_rng(3)).advance(1440)
        loaded = TrafficModel(bursty, TimeConfig(), np.random.default_rng(3)).advance(1440)
        assert loaded.sum() > base.sum()
        assert (loaded > base * 1.4).any()

    def test_loads_never_negative(self):
        cfg = TrafficConfig(noise_sigma_frac=0.5, noise_ar1_rho=0.9)
        loads = TrafficModel(cfg, TimeConfig(), np.random.default_rng(11)).advance(5000)
        assert (loads >= 0).all()
