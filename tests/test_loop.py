"""
Tests for the live control loop (S12): config, advisor observation parity,
action mapping, loop behavior, and never-crash error semantics.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fonpr.actuators import DryRunActuator
from fonpr.advisors.throughput_advisor import ThroughputAdvisor
from fonpr.loop import LoopConfig, action_to_request, build_policy, run_loop
from fonpr.sim import ACTION_LARGE, ACTION_NOOP, ACTION_SMALL

FIXTURES = Path(__file__).parent / "fixtures"


class StubProm:
    """Stands in for PrometheusConnect using recorded fixture responses."""

    def __init__(self, range_result=None, node_result=None):
        self.range_result = (
            json.loads((FIXTURES / "prom_throughput_range.json").read_text())
            if range_result is None
            else range_result
        )
        self.node_result = (
            json.loads((FIXTURES / "prom_upf_node.json").read_text())
            if node_result is None
            else node_result
        )

    def custom_query_range(self, *args, **kwargs):
        return self.range_result

    def custom_query(self, *args, **kwargs):
        return self.node_result


class StubAdvisor:
    """Deterministic observations for loop tests."""

    def __init__(self, throughput, instance="small"):
        self._tput = throughput
        self._instance = instance

    def observe(self, window_minutes, sample_rate_per_minute=1):
        n = window_minutes * sample_rate_per_minute
        large = 1.0 if self._instance == "large" else 0.0
        small = 1.0 if self._instance == "small" else 0.0
        return np.stack(
            [np.full(n, self._tput), np.full(n, large), np.full(n, small)], axis=1
        ).astype(np.float32)


class TestLoopConfig:
    def test_defaults_are_dry_run(self):
        assert LoopConfig().actuator == "dry-run"

    def test_template_loads_and_is_dry_run(self):
        cfg = LoopConfig.from_yaml(Path(__file__).parents[1] / "configs/agent-dryrun.yaml")
        assert cfg.actuator == "dry-run"

    def test_unknown_key_rejected(self, tmp_path):
        path = tmp_path / "loop.yaml"
        path.write_text("revenue: 1\n")
        with pytest.raises(KeyError, match="revenue"):
            LoopConfig.from_yaml(path)

    def test_checkpoint_policy_requires_path(self):
        with pytest.raises(ValueError, match="checkpoint_path"):
            LoopConfig(policy="checkpoint")

    def test_bad_policy_and_actuator_rejected(self):
        with pytest.raises(ValueError):
            LoopConfig(policy="sac")
        with pytest.raises(ValueError):
            LoopConfig(actuator="kubectl")


class TestThroughputAdvisor:
    def make(self, **kwargs):
        return ThroughputAdvisor(
            "prom:9090", "m4.xlarge", "t3.medium", client=StubProm(**kwargs)
        )

    def test_observation_matches_sim_contract(self):
        obs = self.make().observe(window_minutes=15)
        assert obs.shape == (15, 3)
        assert obs.dtype == np.float32
        assert (obs[:, 0] > 0).all()
        # Fixture pins the UPF to t3.medium: small on, large off.
        assert (obs[:, 2] == 1.0).all() and (obs[:, 1] == 0.0).all()

    def test_empty_throughput_raises(self):
        with pytest.raises(RuntimeError, match="throughput"):
            self.make(range_result=[]).observe(15)

    def test_empty_node_labels_raises(self):
        with pytest.raises(RuntimeError, match="node-label"):
            self.make(node_result=[]).observe(15)


class TestActionMapping:
    def test_noop_maps_to_no_actuation(self):
        assert action_to_request(ACTION_NOOP, LoopConfig()) is None

    def test_sizing_actions_map_to_node_selector(self):
        cfg = LoopConfig()
        large = action_to_request(ACTION_LARGE, cfg)
        small = action_to_request(ACTION_SMALL, cfg)
        assert large["nodeSelector"]["node.kubernetes.io/instance-type"] == "m4.xlarge"
        assert small["nodeSelector"]["node.kubernetes.io/instance-type"] == "t3.medium"


class TestRunLoop:
    def test_high_load_triggers_dry_run_actuation(self):
        cfg = LoopConfig(policy="reactive", interval_minutes=0.0)
        # Load near small capacity: reactive scales up immediately.
        advisor = StubAdvisor(throughput=7.9e6, instance="small")
        history = run_loop(cfg, max_iterations=3, advisor=advisor, sleep=lambda _s: None)
        assert len(history) == 3
        assert history[0]["action"] == ACTION_LARGE
        assert history[0]["applied"] is True
        assert history[0]["reference"] == "dry-run"

    def test_low_load_never_actuates(self):
        cfg = LoopConfig(policy="reactive", interval_minutes=0.0)
        advisor = StubAdvisor(throughput=1e6, instance="small")
        history = run_loop(cfg, max_iterations=2, advisor=advisor, sleep=lambda _s: None)
        assert all(r["request"] is None for r in history)
        assert all(r["applied"] is False for r in history)

    def test_advisor_failure_skips_iteration_not_loop(self):
        class FailingAdvisor:
            calls = 0

            def observe(self, *_a, **_k):
                self.calls += 1
                raise RuntimeError("prometheus unreachable")

        cfg = LoopConfig(interval_minutes=0.0)
        advisor = FailingAdvisor()
        history = run_loop(cfg, max_iterations=3, advisor=advisor, sleep=lambda _s: None)
        assert len(history) == 3
        assert all("error" in r for r in history)
        assert advisor.calls == 3

    def test_sleep_called_between_iterations(self):
        sleeps = []
        cfg = LoopConfig(interval_minutes=2.0)
        run_loop(
            cfg,
            max_iterations=3,
            advisor=StubAdvisor(1e6),
            sleep=sleeps.append,
        )
        assert sleeps == [120.0, 120.0]  # no sleep after the final iteration


class TestBuildPolicy:
    def test_all_baseline_names_build(self):
        for name in ("noop", "threshold", "reactive", "forecast"):
            assert build_policy(LoopConfig(policy=name)) is not None

    def test_dry_run_actuator_records(self):
        actuator = DryRunActuator()
        result = actuator.apply({"target_pod": "upf"})
        assert result.success and actuator.history
