"""
Tests for the evaluation harness (S4): metrics accounting, oracle regret
invariants, and the output bundle contract.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from fonpr.eval import EvalConfig, evaluate, scenario_config
from fonpr.eval.report import summarize, write_report

QUICK = EvalConfig(scenarios=("steady", "diurnal_bursty"), n_seeds=2, episode_days=1.0)
ENERGY_QUICK = EvalConfig(scenarios=("diurnal",), n_seeds=2, episode_days=1.0, energy=True)


@pytest.fixture(scope="module")
def quick_results():
    return evaluate(QUICK)


@pytest.fixture(scope="module")
def energy_results():
    return evaluate(ENERGY_QUICK)


def test_grid_shape(quick_results):
    results, timelines = quick_results
    # 4 baselines + oracle, per scenario, per seed.
    assert len(results) == 2 * 2 * 5
    assert set(results.policy) == {"noop", "threshold", "reactive", "forecast", "oracle"}
    assert set(timelines) == {"steady", "diurnal_bursty"}
    for scenario_lines in timelines.values():
        for series in scenario_lines.values():
            assert set(series) == {"offered", "served", "capacity"}


def test_oracle_has_zero_regret_and_lower_bounds_all(quick_results):
    results, _ = quick_results
    oracle = results[results.policy == "oracle"]
    assert np.allclose(oracle.regret_usd, 0.0, atol=1e-6)
    assert (results.regret_usd >= -1e-6).all()


def test_total_cost_is_infra_plus_penalty(quick_results):
    results, _ = quick_results
    np.testing.assert_allclose(
        results.total_cost_usd, results.infra_cost_usd + results.penalty_usd
    )


def test_noop_on_small_has_minimal_infra_cost(quick_results):
    results, _ = quick_results
    for _key, group in results.groupby(["scenario", "seed"]):
        noop_infra = group[group.policy == "noop"].infra_cost_usd.iloc[0]
        assert noop_infra <= group.infra_cost_usd.max() + 1e-9


def test_summarize_layout(quick_results):
    results, _ = quick_results
    summary = summarize(results)
    assert {"scenario", "policy", "total_cost_usd_mean", "total_cost_usd_ci95"} <= set(
        summary.columns
    )
    assert len(summary) == 2 * 5


def test_scenario_configs_differ():
    steady = scenario_config("steady", 1.0)
    bursty = scenario_config("diurnal_bursty", 1.0)
    assert steady.traffic.burst_rate_per_day == 0.0
    assert bursty.traffic.burst_rate_per_day > 0
    with pytest.raises(ValueError):
        scenario_config("nonexistent", 1.0)


def test_report_bundle_contract(tmp_path, quick_results):
    results, timelines = quick_results
    out_dir = write_report(results, timelines, QUICK, tmp_path)
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "results.md").exists()
    assert (out_dir / "run_meta.yaml").exists()
    # run_meta pins all three reproducibility legs: code (SHA), inputs
    # (config), and environment (python + package versions) — S4.3.
    with open(out_dir / "run_meta.yaml", encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    assert set(meta) == {"git_sha", "generated_utc", "eval_config", "environment"}
    assert meta["environment"]["python"]
    assert meta["environment"]["packages"]["numpy"]
    for scenario in QUICK.scenarios:
        assert (out_dir / "plots" / f"cost_{scenario}.png").exists()
        assert (out_dir / "plots" / f"timeline_{scenario}.png").exists()
    # results.csv round-trips with identical totals.
    reloaded = pd.read_csv(out_dir / "results.csv")
    np.testing.assert_allclose(
        reloaded.total_cost_usd.sum(), results.total_cost_usd.sum()
    )


def test_energy_scenario_config():
    energy = scenario_config("diurnal", 1.0, energy=True)
    plain = scenario_config("diurnal", 1.0)
    assert energy.econ.power is not None
    assert plain.econ.power is None
    # Demand is identical across econ variants (S4.2).
    assert energy.traffic == plain.traffic
    assert energy.time == plain.time


def test_energy_eval_grid(energy_results):
    results, _ = energy_results
    # Every policy draws power every tick, so energy is strictly positive
    # everywhere, and total cost still decomposes as infra + penalty.
    assert (results.energy_kwh > 0).all()
    np.testing.assert_allclose(
        results.total_cost_usd, results.infra_cost_usd + results.penalty_usd
    )
    # Oracle DP and env share the costing module; regret stays exact.
    oracle = results[results.policy == "oracle"]
    assert np.allclose(oracle.regret_usd, 0.0, atol=1e-6)
    assert (results.regret_usd >= -1e-6).all()


def test_energy_markdown_gains_energy_column(tmp_path, energy_results, quick_results):
    energy_res, energy_lines = energy_results
    out_dir = write_report(energy_res, energy_lines, ENERGY_QUICK, tmp_path / "energy")
    assert "Energy (kWh)" in (out_dir / "results.md").read_text(encoding="utf-8")
    price_res, price_lines = quick_results
    out_dir = write_report(price_res, price_lines, QUICK, tmp_path / "price")
    assert "Energy (kWh)" not in (out_dir / "results.md").read_text(encoding="utf-8")


def test_eval_config_yaml_roundtrips_energy(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text("n_seeds: 3\nenergy: true\n")
    assert EvalConfig.from_yaml(path).energy is True


def test_eval_config_yaml_rejects_unknown_keys(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text("n_seeds: 3\nrevenue: 1\n")
    with pytest.raises(KeyError, match="revenue"):
        EvalConfig.from_yaml(path)
