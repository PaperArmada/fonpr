"""
B4 MPC baseline tests (S3, ADR-0005 rung 2).

The load-bearing contract is exact-forecast equivalence: fed the true
offered levels with unit margin and a horizon covering the remaining
episode, the receding-horizon planner must reproduce the oracle DP's
action sequence exactly (principle of optimality, identical tie-breaking).
"""

import numpy as np

from fonpr.policies import MpcPolicy, MpcPoolPolicy, make_baselines, plan_oracle_actions
from fonpr.policies.mpc import plan_first_target
from fonpr.policies.oracle import pool_step_cost, step_cost
from fonpr.sim import ACTION_LARGE, ACTION_NOOP, SimConfig
from fonpr.sim.config import PlantConfig, PoolConfig

CFG = SimConfig()
SMALL_CAP = CFG.plant.small_capacity_bytes_per_sec
TICKS = CFG.time.ticks_per_step

POOL_CFG = SimConfig(
    plant=PlantConfig(pool=PoolConfig())  # defaults: t3.medium x 1..8, initial 1
)
POOL = POOL_CFG.plant.pool
NODE_CAP = POOL.node_capacity_bytes_per_sec


def steps_from_levels(levels: list[float]) -> list[np.ndarray]:
    """Constant offered-load arrays, one per step, as the DP consumes them."""
    return [np.full(TICKS, level) for level in levels]


def binary_obs(throughput: float, instance: str) -> np.ndarray:
    n = CFG.time.window_ticks
    large = 1.0 if instance in ("large", "transition") else 0.0
    small = 1.0 if instance in ("small", "transition") else 0.0
    return np.stack(
        [np.full(n, throughput), np.full(n, large), np.full(n, small)], axis=1
    ).astype(np.float32)


def pool_obs(throughput: float, count: int, target: int | None = None) -> np.ndarray:
    n = POOL_CFG.time.window_ticks
    target = count if target is None else target
    return np.stack(
        [
            np.full(n, throughput),
            np.full(n, count / POOL.max_nodes),
            np.full(n, target / POOL.max_nodes),
        ],
        axis=1,
    ).astype(np.float32)


def receding_actions(config: SimConfig, levels: list[float]) -> list:
    """Replay the receding-horizon planner over a known level sequence,
    exactly as MPC would with perfect forecasts and unit margin."""
    pool = config.plant.pool
    if pool is not None:
        states: tuple = tuple(range(pool.min_nodes, pool.max_nodes + 1))
        state = pool.initial_nodes
        cost_fn = lambda s, t, lvl: pool_step_cost(  # noqa: E731
            config, s, t, np.full(TICKS, lvl)
        )
    else:
        states = ("small", "large")
        state = config.plant.initial_instance
        cost_fn = lambda s, t, lvl: step_cost(  # noqa: E731
            config, s, t, np.full(TICKS, lvl)
        )
    out = []
    for t in range(len(levels)):
        state = plan_first_target(states, state, levels[t:], cost_fn)
        out.append(state)
    return out


class TestExactForecastEquivalence:
    """Receding horizon with the true levels == oracle DP, exactly."""

    def test_binary_matches_oracle(self):
        levels = [0.3 * SMALL_CAP] * 3 + [2.5 * SMALL_CAP] * 4 + [0.3 * SMALL_CAP] * 5
        oracle_actions, _ = plan_oracle_actions(steps_from_levels(levels), CFG)
        targets = receding_actions(CFG, levels)
        # Map oracle actions onto realized states for comparison.
        state = CFG.plant.initial_instance
        oracle_targets = []
        for a in oracle_actions:
            if a == ACTION_LARGE:
                state = "large"
            elif a == ACTION_NOOP:
                pass
            else:
                state = "small"
            oracle_targets.append(state)
        assert targets == oracle_targets

    def test_pool_matches_oracle(self):
        levels = (
            [0.5 * NODE_CAP] * 3
            + [3.4 * NODE_CAP] * 4
            + [1.6 * NODE_CAP] * 3
            + [0.4 * NODE_CAP] * 4
        )
        oracle_actions, _ = plan_oracle_actions(steps_from_levels(levels), POOL_CFG)
        oracle_targets = [POOL.min_nodes + a for a in oracle_actions]
        assert receding_actions(POOL_CFG, levels) == oracle_targets


class TestAnticipation:
    """The planner pre-scales during cheap steps ahead of a forecast ramp,
    which no reactive baseline can do — the reason B4 exists."""

    def test_binary_upscales_before_ramp(self):
        low, high = 0.3 * SMALL_CAP, 2.5 * SMALL_CAP
        cost_fn = lambda s, t, lvl: step_cost(CFG, s, t, np.full(TICKS, lvl))  # noqa: E731
        target = plan_first_target(("small", "large"), "small", [low, high, high], cost_fn)
        assert target == "large"  # transition starts under low load, not during the ramp

    def test_pool_upscales_before_ramp(self):
        low, high = 0.5 * NODE_CAP, 3.4 * NODE_CAP
        states = tuple(range(POOL.min_nodes, POOL.max_nodes + 1))
        cost_fn = lambda s, t, lvl: pool_step_cost(  # noqa: E731
            POOL_CFG, s, t, np.full(TICKS, lvl)
        )
        target = plan_first_target(states, POOL.min_nodes, [low, high, high], cost_fn)
        assert target >= 4  # capacity ready for 3.4x nodes before the ramp arrives

    def test_no_upscale_when_future_stays_low(self):
        low = 0.3 * SMALL_CAP
        cost_fn = lambda s, t, lvl: step_cost(CFG, s, t, np.full(TICKS, lvl))  # noqa: E731
        assert plan_first_target(("small", "large"), "small", [low] * 8, cost_fn) == "small"

    def test_margin_escapes_saturation(self):
        """Served-history forecasts are self-fulfilling under saturation
        (capacity always covers yesterday's served), so the safety margin is
        the escape mechanism — for B4 exactly as for B3. A saturated count-1
        history with margin > 1 must force a scale-up."""
        states = tuple(range(POOL.min_nodes, POOL.max_nodes + 1))
        cost_fn = lambda s, t, lvl: pool_step_cost(  # noqa: E731
            POOL_CFG, s, t, np.full(TICKS, lvl)
        )
        saturated = [1.0 * NODE_CAP * 1.15] * 4  # margined saturated forecast
        assert plan_first_target(states, 1, saturated, cost_fn) >= 2
        unmargined = [1.0 * NODE_CAP] * 4
        assert plan_first_target(states, 1, unmargined, cost_fn) == 1  # the trap


class TestPolicyBehavior:
    def test_holds_during_transition_binary(self):
        p = MpcPolicy(CFG)
        assert p.act(binary_obs(0.3 * SMALL_CAP, "transition")) == ACTION_NOOP

    def test_holds_during_transition_pool(self):
        p = MpcPoolPolicy(POOL_CFG)
        action = p.act(pool_obs(0.5 * NODE_CAP, count=2, target=3))
        assert action == POOL.action_of_count(2)  # holds the current count

    def test_deterministic_across_reset(self):
        seq = [0.3, 0.5, 2.0, 2.2, 0.4, 0.3]
        p = MpcPolicy(CFG)
        first = [p.act(binary_obs(x * SMALL_CAP, "small")) for x in seq]
        p.reset()
        second = [p.act(binary_obs(x * SMALL_CAP, "small")) for x in seq]
        assert first == second

    def test_make_baselines_includes_mpc_last(self):
        for cfg in (CFG, POOL_CFG):
            names = [p.name for p in make_baselines(cfg)]
            assert names == ["noop", "threshold", "reactive", "forecast", "mpc"]


class TestEndToEnd:
    """Deterministic diurnal invariants. Which policy wins a scenario is an
    empirical question for the eval harness (recorded campaigns), not a
    unit-test assertion; what is pinned here is that B4 stays sane: it
    tracks the cycle, bounds regret, and lands near B3 (identical
    forecaster and margin — measured 2026-08-04: a ~0.6% conservatism
    premium from pricing violations against margined levels)."""

    def test_pool_mpc_clean_diurnal_invariants(self):
        from fonpr.eval.harness import run_episode
        from fonpr.policies import ForecastPoolPolicy, NoopPoolPolicy
        from fonpr.sim import FONPRSimEnv
        from fonpr.sim.config import PlantConfig, TimeConfig, TrafficConfig

        cfg = SimConfig(
            time=TimeConfig(episode_days=3.0),
            traffic=TrafficConfig(noise_sigma_frac=0.0, burst_rate_per_day=0.0),
            plant=PlantConfig(pool=PoolConfig()),
        )
        env = FONPRSimEnv(cfg)
        mpc, offered, _ = run_episode(env, MpcPoolPolicy(cfg), seed=0)
        fc, _, _ = run_episode(env, ForecastPoolPolicy(cfg), seed=0)
        noop, _, _ = run_episode(env, NoopPoolPolicy(cfg), seed=0)
        _, oracle_cost = plan_oracle_actions(offered, cfg)

        assert mpc["total_cost_usd"] >= oracle_cost - 1e-9  # oracle stays a bound
        assert mpc["total_cost_usd"] < 0.1 * noop["total_cost_usd"]  # tracks the cycle
        assert mpc["action_churn"] > 0
        # Near-parity with B3 on the scenario where both share every input.
        assert mpc["total_cost_usd"] <= fc["total_cost_usd"] * 1.02
