"""
Tests for baseline policies and the oracle (S3), over hand-built
observations with known-correct decisions.
"""

import numpy as np
import pytest

from fonpr.policies import (
    ForecastPolicy,
    NoopPolicy,
    OraclePolicy,
    ReactivePolicy,
    ThresholdPolicy,
    plan_oracle_actions,
)
from fonpr.sim import ACTION_LARGE, ACTION_NOOP, ACTION_SMALL, SimConfig
from fonpr.sim.config import TrafficConfig

CFG = SimConfig()
SMALL_CAP = CFG.plant.small_capacity_bytes_per_sec
LARGE_CAP = CFG.plant.large_capacity_bytes_per_sec


def obs(throughput: float, instance: str) -> np.ndarray:
    """Build a steady observation window on the given instance."""
    n = CFG.time.window_ticks
    large = 1.0 if instance in ("large", "transition") else 0.0
    small = 1.0 if instance in ("small", "transition") else 0.0
    return np.stack(
        [np.full(n, throughput), np.full(n, large), np.full(n, small)], axis=1
    ).astype(np.float32)


class TestNoop:
    def test_never_acts(self):
        policy = NoopPolicy()
        assert policy.act(obs(1e9, "small")) == ACTION_NOOP


class TestThreshold:
    def test_scales_up_after_patience(self):
        policy = ThresholdPolicy(CFG, up_patience=2)
        high = obs(0.9 * SMALL_CAP, "small")
        assert policy.act(high) == ACTION_NOOP  # patience 1 of 2
        assert policy.act(high) == ACTION_LARGE

    def test_low_load_resets_patience(self):
        policy = ThresholdPolicy(CFG, up_patience=2)
        assert policy.act(obs(0.9 * SMALL_CAP, "small")) == ACTION_NOOP
        assert policy.act(obs(0.1 * SMALL_CAP, "small")) == ACTION_NOOP
        assert policy.act(obs(0.9 * SMALL_CAP, "small")) == ACTION_NOOP  # counter restarted

    def test_scales_down_after_patience_and_cooldown_blocks(self):
        policy = ThresholdPolicy(CFG, down_patience=2, cooldown_steps=3)
        low = obs(0.2 * SMALL_CAP, "large")
        assert policy.act(low) == ACTION_NOOP
        assert policy.act(low) == ACTION_SMALL
        # Cooldown: even sustained high load can't trigger for 3 steps.
        high = obs(0.9 * SMALL_CAP, "small")
        assert [policy.act(high) for _ in range(3)] == [ACTION_NOOP] * 3

    def test_holds_during_transition(self):
        policy = ThresholdPolicy(CFG)
        assert policy.act(obs(1e9, "transition")) == ACTION_NOOP


class TestReactive:
    def test_up_when_above_target_utilization(self):
        policy = ReactivePolicy(CFG, target_utilization=0.7)
        assert policy.act(obs(0.71 * SMALL_CAP, "small")) == ACTION_LARGE
        assert policy.act(obs(0.69 * SMALL_CAP, "small")) == ACTION_NOOP

    def test_down_when_fits_small_comfortably(self):
        policy = ReactivePolicy(CFG, target_utilization=0.7)
        assert policy.act(obs(0.5 * SMALL_CAP, "large")) == ACTION_SMALL
        assert policy.act(obs(0.9 * SMALL_CAP, "large")) == ACTION_NOOP


class TestForecast:
    def test_uses_yesterday_same_step(self):
        policy = ForecastPolicy(CFG, safety_margin=1.15)
        steps_per_day = int(24 * 60 / CFG.time.step_minutes)
        # Day 1: low load on small — no action.
        low = obs(0.3 * SMALL_CAP, "small")
        for _ in range(steps_per_day):
            assert policy.act(low) == ACTION_NOOP
        # Day 2 starts: yesterday-same-step was low, so keep small even
        # though the instantaneous reading is now high.
        assert policy.act(obs(0.95 * SMALL_CAP, "small")) == ACTION_NOOP

    def test_saturated_yesterday_demands_large(self):
        policy = ForecastPolicy(CFG, safety_margin=1.15)
        policy._history = [SMALL_CAP] * int(24 * 60 / CFG.time.step_minutes)
        assert policy.act(obs(0.5 * SMALL_CAP, "small")) == ACTION_LARGE

    def test_scales_down_when_forecast_fits(self):
        policy = ForecastPolicy(CFG, safety_margin=1.15)
        assert policy.act(obs(0.3 * SMALL_CAP, "large")) == ACTION_SMALL


class TestOracle:
    def n_steps(self, n, load):
        ticks = CFG.time.ticks_per_step
        return [np.full(ticks, load) for _ in range(n)]

    def test_constant_low_load_stays_small(self):
        actions, cost = plan_oracle_actions(self.n_steps(8, 0.5 * SMALL_CAP), CFG)
        assert actions == [ACTION_NOOP] * 8
        assert cost == pytest.approx(0.0416 * 8 * CFG.time.step_minutes / 60)

    def test_constant_high_load_goes_large_once(self):
        actions, _ = plan_oracle_actions(self.n_steps(8, 0.9 * LARGE_CAP), CFG)
        assert actions[0] == ACTION_LARGE
        assert actions[1:] == [ACTION_NOOP] * 7

    def test_brief_dip_not_worth_transitioning(self):
        # High load with a single low step: two transitions cost more than
        # staying large for one step.
        steps = (
            self.n_steps(3, 0.9 * LARGE_CAP)
            + self.n_steps(1, 0.1 * SMALL_CAP)
            + self.n_steps(3, 0.9 * LARGE_CAP)
        )
        actions, _ = plan_oracle_actions(steps, CFG)
        assert actions[0] == ACTION_LARGE
        assert ACTION_SMALL not in actions[1:]

    def test_oracle_cost_is_a_lower_bound_in_the_env(self):
        # Roll a NOOP episode to capture offered load, plan, replay the plan,
        # and check the realized cost matches the DP's own accounting.
        from fonpr.sim import FONPRSimEnv
        from fonpr.sim.config import TimeConfig

        cfg = SimConfig(time=TimeConfig(episode_days=1.0), traffic=TrafficConfig())
        env = FONPRSimEnv(cfg)
        env.reset(seed=42)
        offered_steps, noop_cost = [], 0.0
        for _ in range(cfg.time.episode_steps):
            _, _, _, _, info = env.step(ACTION_NOOP)
            offered_steps.append(info["offered_series"])
            noop_cost += info["step_cost_usd"] + info["step_penalty_usd"]

        actions, planned_cost = plan_oracle_actions(offered_steps, cfg)
        env.reset(seed=42)
        oracle, realized = OraclePolicy(actions), 0.0
        obs_, _ = env.reset(seed=42)
        for _ in range(cfg.time.episode_steps):
            obs_, _, _, _, info = env.step(oracle.act(obs_))
            realized += info["step_cost_usd"] + info["step_penalty_usd"]

        assert realized == pytest.approx(planned_cost, rel=1e-9)
        assert realized <= noop_cost + 1e-9
