"""
S4.5 robustness protocol tests (ADR-0005 rung 3).

The load-bearing contract: with no perturbations (or an all-defaults cell)
the S4.2 protocol is unchanged bit for bit; with a cell, only the WORLD
changes while policies keep their nominal beliefs.
"""

import numpy as np
import pytest

from fonpr.eval import EvalConfig, PerturbConfig, evaluate, perturbed_world, scenario_config

QUICK = EvalConfig(scenarios=("diurnal",), n_seeds=2, episode_days=1.0)
IDENTITY_CELL = EvalConfig(
    scenarios=("diurnal",),
    n_seeds=2,
    episode_days=1.0,
    perturbations={"nominal": PerturbConfig()},
)


class TestIdentity:
    def test_all_defaults_cell_is_bit_identical_to_no_perturbations(self):
        plain, _ = evaluate(QUICK)
        celled, _ = evaluate(IDENTITY_CELL)
        assert list(plain.policy) == list(celled.policy)
        assert (celled.perturb == "nominal").all()
        for metric in ("total_cost_usd", "violation_minutes", "action_churn", "regret_usd"):
            np.testing.assert_array_equal(plain[metric].values, celled[metric].values)

    def test_determinism_across_runs_with_noise(self):
        cfg = EvalConfig(
            scenarios=("diurnal",),
            n_seeds=2,
            episode_days=1.0,
            perturbations={"noisy": PerturbConfig(obs_noise_sigma_frac=0.05)},
        )
        a, _ = evaluate(cfg)
        b, _ = evaluate(cfg)
        np.testing.assert_array_equal(a.total_cost_usd.values, b.total_cost_usd.values)


class TestWorldPerturbation:
    def test_capacity_scale_binary_and_pool(self):
        binary = scenario_config("diurnal", 1.0)
        world = perturbed_world(binary, PerturbConfig(capacity_scale=0.5))
        assert world.plant.small_capacity_bytes_per_sec == pytest.approx(
            0.5 * binary.plant.small_capacity_bytes_per_sec
        )
        assert world.plant.large_capacity_bytes_per_sec == pytest.approx(
            0.5 * binary.plant.large_capacity_bytes_per_sec
        )
        pool = scenario_config("diurnal", 1.0, pool=True)
        world = perturbed_world(pool, PerturbConfig(capacity_scale=1.3, lag_scale=3.0))
        assert world.plant.pool.node_capacity_bytes_per_sec == pytest.approx(
            1.3 * pool.plant.pool.node_capacity_bytes_per_sec
        )
        assert world.plant.pool.transition_lag_minutes == 15

    def test_belief_unchanged(self):
        belief = scenario_config("diurnal", 1.0, pool=True)
        perturbed_world(belief, PerturbConfig(capacity_scale=0.7))
        # Frozen dataclasses guarantee this, but the contract is worth pinning:
        # the belief object the policies are built from is untouched.
        assert belief.plant.pool.node_capacity_bytes_per_sec == 8.0e6

    def test_smaller_world_capacity_hurts_noop(self):
        base = EvalConfig(scenarios=("diurnal",), n_seeds=2, episode_days=1.0)
        shrunk = EvalConfig(
            scenarios=("diurnal",),
            n_seeds=2,
            episode_days=1.0,
            perturbations={"cap50": PerturbConfig(capacity_scale=0.5)},
        )
        nominal, _ = evaluate(base)
        world, _ = evaluate(shrunk)
        noop_nominal = nominal[nominal.policy == "noop"].violation_minutes.mean()
        noop_world = world[world.policy == "noop"].violation_minutes.mean()
        assert noop_world > noop_nominal


class TestConfigParsing:
    def test_yaml_round_trip(self, tmp_path):
        p = tmp_path / "eval.yaml"
        p.write_text(
            "scenarios: [diurnal]\n"
            "n_seeds: 2\n"
            "perturbations:\n"
            "  cap-30: {capacity_scale: 0.7}\n"
            "  noisy: {obs_noise_sigma_frac: 0.05}\n"
        )
        cfg = EvalConfig.from_yaml(p)
        assert cfg.perturbations["cap-30"].capacity_scale == 0.7
        assert cfg.perturbations["noisy"].obs_noise_sigma_frac == 0.05

    def test_unknown_cell_key_rejected(self, tmp_path):
        p = tmp_path / "eval.yaml"
        p.write_text("perturbations:\n  bad: {capacity_pct: 30}\n")
        with pytest.raises(KeyError):
            EvalConfig.from_yaml(p)

    def test_unsafe_label_rejected(self, tmp_path):
        p = tmp_path / "eval.yaml"
        p.write_text("perturbations:\n  'a b/c': {}\n")
        with pytest.raises(ValueError):
            EvalConfig.from_yaml(p)

    def test_invalid_values_rejected(self):
        with pytest.raises(ValueError):
            PerturbConfig(capacity_scale=0.0)
        with pytest.raises(ValueError):
            PerturbConfig(obs_noise_sigma_frac=-0.1)
