"""
Pool plant variant tests (S14, ADR-0004): config validation, resize
economics, env contract, generalized baselines, and oracle consistency.
"""

import numpy as np
import pytest

from fonpr.eval import EvalConfig, evaluate, scenario_config
from fonpr.policies import (
    ForecastPoolPolicy,
    MpcPoolPolicy,
    NoopPoolPolicy,
    ReactivePoolPolicy,
    ThresholdPoolPolicy,
    make_baselines,
)
from fonpr.sim import (
    EconConfig,
    FONPRSimEnv,
    PlantConfig,
    PoolConfig,
    PoolPlantModel,
    SimConfig,
    TimeConfig,
)


def pool_sim(episode_days: float = 0.5, **pool_kwargs) -> SimConfig:
    return SimConfig(
        time=TimeConfig(episode_days=episode_days),
        plant=PlantConfig(pool=PoolConfig(**pool_kwargs)),
    )


class TestPoolConfig:
    def test_action_count_mapping_roundtrips(self):
        pool = PoolConfig(min_nodes=2, max_nodes=6, initial_nodes=2)
        assert pool.n_actions == 5
        for count in range(2, 7):
            assert pool.count_of_action(pool.action_of_count(count)) == count
        with pytest.raises(ValueError):
            pool.count_of_action(5)
        with pytest.raises(ValueError):
            pool.action_of_count(7)

    def test_validation(self):
        with pytest.raises(ValueError):
            PoolConfig(min_nodes=0)
        with pytest.raises(ValueError):
            PoolConfig(min_nodes=3, max_nodes=3)
        with pytest.raises(ValueError):
            PoolConfig(initial_nodes=9)
        with pytest.raises(ValueError):
            PoolConfig(node_capacity_bytes_per_sec=0.0)

    def test_yaml_roundtrip_and_pricing_check(self, tmp_path):
        path = tmp_path / "pool.yaml"
        path.write_text("plant:\n  pool:\n    max_nodes: 4\n    initial_nodes: 2\n")
        cfg = SimConfig.from_yaml(path)
        assert cfg.plant.pool.max_nodes == 4
        assert cfg.plant.pool.initial_nodes == 2
        # Unknown node types fail fast against the pricing table.
        with pytest.raises(KeyError):
            SimConfig(plant=PlantConfig(pool=PoolConfig(node_type="z9.mega")))


class TestPoolPlantModel:
    def make(self, **kwargs):
        pool = PoolConfig(**kwargs)
        time_cfg = TimeConfig()
        return pool, PoolPlantModel(pool, EconConfig(), time_cfg)

    def test_scale_up_capacity_and_billing(self):
        pool, plant = self.make(initial_nodes=2)
        assert plant.request_target(4)
        offered = np.full(15, 1.0e6)
        result = plant.advance(offered)
        c = pool.node_capacity_bytes_per_sec
        # Lag is 5 ticks: old capacity, new (max) billing during boot.
        np.testing.assert_allclose(result.capacity[:5], 2 * c)
        np.testing.assert_allclose(result.billed_count[:5], 4)
        np.testing.assert_allclose(result.active_count[:5], 2)
        np.testing.assert_allclose(result.target_count[:5], 4)
        np.testing.assert_allclose(result.capacity[5:], 4 * c)
        np.testing.assert_allclose(result.billed_count[5:], 4)
        assert not plant.in_transition

    def test_scale_down_drains_at_old_capacity_and_billing(self):
        pool, plant = self.make(initial_nodes=5)
        assert plant.request_target(2)
        result = plant.advance(np.full(15, 1.0e6))
        c = pool.node_capacity_bytes_per_sec
        np.testing.assert_allclose(result.capacity[:5], 5 * c)
        np.testing.assert_allclose(result.billed_count[:5], 5)
        np.testing.assert_allclose(result.capacity[5:], 2 * c)
        np.testing.assert_allclose(result.billed_count[5:], 2)

    def test_requests_ignored_during_resize_and_noop_target(self):
        _, plant = self.make(initial_nodes=2)
        assert not plant.request_target(2)  # already there
        assert plant.request_target(3)
        assert not plant.request_target(5)  # in flight
        with pytest.raises(ValueError):
            plant.request_target(99)

    def test_zero_lag_is_immediate(self):
        _, plant = self.make(initial_nodes=1, transition_lag_minutes=0)
        assert plant.request_target(3)
        assert not plant.in_transition
        assert plant.active == 3


class TestPoolEnv:
    def test_spaces_and_observation_columns(self):
        env = FONPRSimEnv(pool_sim())
        assert env.action_space.n == 8
        obs, info = env.reset(seed=0)
        assert obs.shape == (15, 3)
        # Initial count 1 of max 8 in both count and target columns.
        np.testing.assert_allclose(obs[:, 1], 1 / 8)
        np.testing.assert_allclose(obs[:, 2], 1 / 8)
        assert info["node_count"] == 1
        assert info["instance_type"] == "t3.medium"

    def test_resize_applied_and_noop_semantics(self):
        env = FONPRSimEnv(pool_sim())
        env.reset(seed=0)
        pool = env.config.plant.pool
        obs, _, _, _, info = env.step(pool.action_of_count(1))  # current count
        assert info["action_applied"] is False
        obs, _, _, _, info = env.step(pool.action_of_count(4))
        assert info["action_applied"] is True
        assert info["node_count"] == 4  # lag < step, so it lands in-step
        np.testing.assert_allclose(obs[-1, 1], 4 / 8)

    def test_determinism_same_seed(self):
        rewards = []
        for _ in range(2):
            env = FONPRSimEnv(pool_sim())
            env.reset(seed=123)
            total, truncated = 0.0, False
            while not truncated:
                _, r, _, truncated, _ = env.step(3)
                total += r
            rewards.append(total)
        assert rewards[0] == rewards[1]

    def test_time_features_append_after_count_columns(self):
        cfg = SimConfig(
            time=TimeConfig(episode_days=0.5, include_time_features=True),
            plant=PlantConfig(pool=PoolConfig()),
        )
        env = FONPRSimEnv(cfg)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (15, 5)
        np.testing.assert_allclose(obs[:, 1], 1 / 8)


class TestPoolBaselines:
    def make_obs(self, tput: float, count: int, target: int | None = None):
        pool = PoolConfig()
        rows = np.zeros((15, 3), dtype=np.float32)
        rows[:, 0] = tput
        rows[:, 1] = count / pool.max_nodes
        rows[:, 2] = (target if target is not None else count) / pool.max_nodes
        return rows

    def test_make_baselines_selects_pool_variants(self):
        policies = make_baselines(pool_sim())
        assert [type(p) for p in policies] == [
            NoopPoolPolicy,
            ThresholdPoolPolicy,
            ReactivePoolPolicy,
            ForecastPoolPolicy,
            MpcPoolPolicy,
        ]
        assert [p.name for p in policies] == [
            "noop",
            "threshold",
            "reactive",
            "forecast",
            "mpc",
        ]

    def test_noop_holds_current_count(self):
        policy = NoopPoolPolicy(pool_sim())
        assert policy.act(self.make_obs(1e9, 5)) == PoolConfig().action_of_count(5)

    def test_reactive_is_the_hpa_formula(self):
        policy = ReactivePoolPolicy(pool_sim())
        pool = PoolConfig()
        # 14e6 / (0.7 * 8e6) = 2.5 -> 3 nodes.
        assert policy.act(self.make_obs(14e6, 1)) == pool.action_of_count(3)
        # Holds during a resize.
        assert policy.act(self.make_obs(14e6, 1, target=3)) == pool.action_of_count(1)
        # Clips to max_nodes.
        assert policy.act(self.make_obs(1e9, 8)) == pool.action_of_count(8)

    def test_threshold_steps_by_one_with_patience(self):
        policy = ThresholdPoolPolicy(pool_sim(), up_patience=2, cooldown_steps=0)
        pool = PoolConfig()
        saturated = self.make_obs(7.9e6, 1)  # util ~0.99 at one node
        assert policy.act(saturated) == pool.action_of_count(1)  # patience 1/2
        assert policy.act(saturated) == pool.action_of_count(2)  # steps up one

    def test_forecast_sizes_count_to_forecast(self):
        policy = ForecastPoolPolicy(pool_sim())
        pool = PoolConfig()
        # Warmup history short: forecast = current tput; 20e6 * 1.15 / 8e6 -> 3.
        assert policy.act(self.make_obs(20e6, 4)) == pool.action_of_count(3)


POOL_QUICK = EvalConfig(scenarios=("diurnal",), n_seeds=2, episode_days=1.0, pool=True)
POOL_ENERGY_QUICK = EvalConfig(
    scenarios=("diurnal",), n_seeds=2, episode_days=1.0, pool=True, energy=True
)


@pytest.mark.parametrize("eval_cfg", [POOL_QUICK, POOL_ENERGY_QUICK])
def test_pool_eval_oracle_consistency(eval_cfg):
    # evaluate() asserts planned-vs-realized oracle cost to 1e-6 internally;
    # completing without tripping it IS the consistency check (both econs).
    results, _ = evaluate(eval_cfg)
    oracle = results[results.policy == "oracle"]
    assert np.allclose(oracle.regret_usd, 0.0, atol=1e-6)
    assert (results.regret_usd >= -1e-6).all()
    if eval_cfg.energy:
        assert (results.energy_kwh > 0).all()


def test_pool_scenario_config():
    cfg = scenario_config("diurnal", 1.0, pool=True)
    assert cfg.plant.pool is not None
    assert scenario_config("diurnal", 1.0).plant.pool is None
