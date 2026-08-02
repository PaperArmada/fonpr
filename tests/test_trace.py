"""
Tests for trace replay (S1.6): schema validation, tick resampling, gap
rejection, and replay through the env.
"""

import numpy as np
import pandas as pd
import pytest

from fonpr.sim import FONPRSimEnv, SimConfig
from fonpr.sim.config import TimeConfig, TrafficConfig
from fonpr.sim.trace import ReplayTrafficModel, load_trace, resample_to_ticks


def make_trace(tmp_path, minutes=120, period_s=60, value=5e6, fmt="parquet"):
    timestamps = pd.date_range(
        "2026-01-01", periods=minutes * 60 // period_s, freq=f"{period_s}s", tz="UTC"
    )
    df = pd.DataFrame(
        {"timestamp": timestamps, "throughput_bytes_per_sec": np.full(len(timestamps), value)}
    )
    path = tmp_path / f"trace.{ 'csv' if fmt == 'csv' else 'parquet'}"
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return path


class TestLoadAndResample:
    def test_parquet_round_trip(self, tmp_path):
        path = make_trace(tmp_path)
        trace = load_trace(path)
        ticks = resample_to_ticks(trace, TimeConfig())
        assert len(ticks) == 120  # one per minute at sample_rate 1
        np.testing.assert_allclose(ticks, 5e6)

    def test_csv_import(self, tmp_path):
        path = make_trace(tmp_path, fmt="csv")
        assert len(load_trace(path)) > 0

    def test_missing_column_rejected(self, tmp_path):
        path = tmp_path / "bad.parquet"
        pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="60s")}).to_parquet(
            path
        )
        with pytest.raises(ValueError, match="missing column"):
            load_trace(path)

    def test_long_gap_rejected(self, tmp_path):
        timestamps = pd.to_datetime(
            ["2026-01-01 00:00", "2026-01-01 00:01", "2026-01-01 00:10"], utc=True
        )
        path = tmp_path / "gap.parquet"
        pd.DataFrame(
            {"timestamp": timestamps, "throughput_bytes_per_sec": [1.0, 2.0, 3.0]}
        ).to_parquet(path)
        with pytest.raises(ValueError, match="gap"):
            resample_to_ticks(load_trace(path), TimeConfig())

    def test_interpolation_between_sparse_samples(self, tmp_path):
        timestamps = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:02"], utc=True)
        path = tmp_path / "sparse.parquet"
        pd.DataFrame(
            {"timestamp": timestamps, "throughput_bytes_per_sec": [0.0, 2.0]}
        ).to_parquet(path)
        ticks = resample_to_ticks(load_trace(path), TimeConfig())
        np.testing.assert_allclose(ticks, [0.0, 1.0, 2.0])


class TestReplay:
    def test_exhaustion_is_loud(self):
        model = ReplayTrafficModel(np.arange(10.0))
        model.advance(8)
        with pytest.raises(RuntimeError, match="exhausted"):
            model.advance(3)

    def test_env_replays_trace(self, tmp_path):
        # 1 sim-day episode + warm-up window needs > 1 day of trace.
        path = make_trace(tmp_path, minutes=26 * 60, value=5e6)
        cfg = SimConfig(
            time=TimeConfig(episode_days=1.0),
            traffic=TrafficConfig(trace_path=str(path)),
        )
        env = FONPRSimEnv(cfg)
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        np.testing.assert_allclose(info["offered_series"], 5e6)
        # Deterministic regardless of seed: it's a tape.
        env2 = FONPRSimEnv(cfg)
        env2.reset(seed=99)
        _, _, _, _, info2 = env2.step(0)
        np.testing.assert_allclose(info2["offered_series"], info["offered_series"])
