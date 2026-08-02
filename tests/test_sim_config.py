"""
Tests for the sim config (S1.5): defaults, validation, strict YAML loading.
"""

import dataclasses

import pytest

from fonpr.sim import SimConfig
from fonpr.sim.config import EconConfig, TimeConfig, TrafficConfig


def test_defaults_are_valid_and_frozen():
    cfg = SimConfig()
    assert cfg.time.episode_steps == 672  # 7 days at 15-minute steps
    assert cfg.time.window_ticks == 15
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.time.step_minutes = 5


def test_capacity_calibration_matches_adr_0001_d7():
    cfg = SimConfig()
    peak = cfg.traffic.peak_diurnal_load
    assert cfg.plant.large_capacity_bytes_per_sec == pytest.approx(1.2 * peak)
    assert cfg.plant.small_capacity_bytes_per_sec == pytest.approx(0.4 * peak)


def test_penalty_per_minute_matches_adr_0001_d5():
    econ = EconConfig()
    assert econ.penalty_per_violation_minute_usd == pytest.approx(20.0 * 0.20 / 60.0)


def test_yaml_round_trip(tmp_path):
    cfg = SimConfig()
    path = tmp_path / "sim.yaml"
    cfg.to_yaml(path)
    loaded = SimConfig.from_yaml(path)
    assert loaded == cfg


def test_partial_yaml_overrides_only_named_keys(tmp_path):
    path = tmp_path / "sim.yaml"
    path.write_text("traffic:\n  base_load_bytes_per_sec: 1000.0\n")
    cfg = SimConfig.from_yaml(path)
    assert cfg.traffic.base_load_bytes_per_sec == 1000.0
    assert cfg.time == TimeConfig()  # untouched section keeps defaults


def test_unknown_key_is_an_error(tmp_path):
    path = tmp_path / "sim.yaml"
    path.write_text("traffic:\n  revenue_per_byte: 1.0\n")
    with pytest.raises(KeyError, match="revenue_per_byte"):
        SimConfig.from_yaml(path)


def test_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        TimeConfig(step_minutes=0)
    with pytest.raises(ValueError):
        TrafficConfig(diurnal_amplitude=1.5)
    with pytest.raises(ValueError):
        EconConfig(slo_target=0.0)


def test_missing_pricing_entry_fails_fast():
    from fonpr.sim.config import PlantConfig

    with pytest.raises(KeyError):
        SimConfig(plant=PlantConfig(large_instance_type="m5.24xlarge"))
