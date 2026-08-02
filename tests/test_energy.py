"""
Tests for the energy variant (S13): power config validation, watts math,
price-model equivalence when disabled, and oracle/env consistency under
the energy cost model.
"""

import pytest

from fonpr.sim import (
    ACTION_LARGE,
    ACTION_NOOP,
    EconConfig,
    FONPRSimEnv,
    PowerConfig,
    SimConfig,
    TimeConfig,
    TrafficConfig,
)

QUIET = dict(diurnal_amplitude=0.0, noise_sigma_frac=0.0, burst_rate_per_day=0.0)


def energy_config(**traffic) -> SimConfig:
    return SimConfig(
        time=TimeConfig(episode_days=0.5),
        traffic=TrafficConfig(**{**QUIET, **traffic}),
        econ=EconConfig(power=PowerConfig()),
    )


class TestPowerConfig:
    def test_defaults_valid(self):
        power = PowerConfig()
        assert power.watts("t3.medium", 0.0) == 60.0
        assert power.watts("t3.medium", 1.0) == 120.0
        assert power.max_hourly_cost_usd == pytest.approx(280 * 0.12 / 1000)

    def test_idle_above_max_rejected(self):
        with pytest.raises(ValueError, match="idle_watts <= max_watts"):
            PowerConfig(
                idle_watts={"a": 100.0}, max_watts={"a": 50.0}
            )

    def test_mismatched_tables_rejected(self):
        with pytest.raises(ValueError, match="same instance types"):
            PowerConfig(idle_watts={"a": 1.0}, max_watts={"b": 2.0})

    def test_sim_config_requires_power_coverage(self):
        with pytest.raises(KeyError, match="power model"):
            SimConfig(
                econ=EconConfig(
                    power=PowerConfig(
                        idle_watts={"other": 1.0}, max_watts={"other": 2.0}
                    )
                )
            )

    def test_penalty_anchors_to_max_power_cost(self):
        econ = EconConfig(power=PowerConfig())
        expected = 20.0 * (280 * 0.12 / 1000) / 60.0
        assert econ.penalty_per_violation_minute_usd == pytest.approx(expected)


class TestDisabledModelUnchanged:
    def test_price_model_reward_and_zero_energy(self):
        cfg = SimConfig(
            time=TimeConfig(episode_days=0.5),
            traffic=TrafficConfig(base_load_bytes_per_sec=1e6, **QUIET),
        )
        env = FONPRSimEnv(cfg)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(ACTION_NOOP)
        assert info["step_energy_wh"] == 0.0
        assert reward == pytest.approx(-0.0416 * 15 / 60)  # unchanged S2 economics


class TestEnergyAccounting:
    def test_steady_small_load_energy_math(self):
        # Constant 4 MB/s on the small tier (cap 8 MB/s): utilization 0.5.
        cfg = energy_config(base_load_bytes_per_sec=4e6)
        env = FONPRSimEnv(cfg)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(ACTION_NOOP)
        watts = 60 + (120 - 60) * 0.5
        expected_wh = watts * 15 / 60
        assert info["step_energy_wh"] == pytest.approx(expected_wh, rel=1e-6)
        expected_cost = expected_wh / 1000 * 0.12
        assert info["step_cost_usd"] == pytest.approx(expected_cost, rel=1e-6)
        assert reward == pytest.approx(-expected_cost, rel=1e-6)

    def test_transition_adds_idle_draw_of_second_node(self):
        cfg = energy_config(base_load_bytes_per_sec=4e6)
        env = FONPRSimEnv(cfg)
        env.reset(seed=0)
        _, _, _, _, info = env.step(ACTION_LARGE)
        # 5 transition minutes: small serves at util 0.5 while large idles;
        # 10 minutes: large serves at util 4/24.
        lag_wh = (60 + 60 * 0.5 + 120) * 5 / 60
        after_watts = 120 + (280 - 120) * (4e6 / 24e6)
        after_wh = after_watts * 10 / 60
        assert info["step_energy_wh"] == pytest.approx(lag_wh + after_wh, rel=1e-6)

    def test_energy_scales_with_load(self):
        low = FONPRSimEnv(energy_config(base_load_bytes_per_sec=1e6))
        high = FONPRSimEnv(energy_config(base_load_bytes_per_sec=7e6))
        low.reset(seed=0)
        high.reset(seed=0)
        _, _, _, _, info_low = low.step(ACTION_NOOP)
        _, _, _, _, info_high = high.step(ACTION_NOOP)
        assert info_high["step_energy_wh"] > info_low["step_energy_wh"]


class TestOracleConsistencyUnderEnergyModel:
    def test_planned_cost_matches_realized(self):
        from fonpr.policies import OraclePolicy, plan_oracle_actions

        cfg = SimConfig(
            time=TimeConfig(episode_days=1.0),
            econ=EconConfig(power=PowerConfig()),
        )
        env = FONPRSimEnv(cfg)
        env.reset(seed=11)
        offered_steps = []
        for _ in range(cfg.time.episode_steps):
            _, _, _, _, info = env.step(ACTION_NOOP)
            offered_steps.append(info["offered_series"])

        actions, planned = plan_oracle_actions(offered_steps, cfg)
        obs, _ = env.reset(seed=11)
        oracle, realized = OraclePolicy(actions), 0.0
        for _ in range(cfg.time.episode_steps):
            obs, _, _, _, info = env.step(oracle.act(obs))
            realized += info["step_cost_usd"] + info["step_penalty_usd"]
        assert realized == pytest.approx(planned, rel=1e-9)
