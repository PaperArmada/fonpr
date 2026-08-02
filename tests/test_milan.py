"""
Tests for the Milan dataset converter, over synthetic fixtures in the
source TSV format (square_id, time_interval_ms, country_code, sms_in,
sms_out, call_in, call_out, internet_traffic).
"""

import pandas as pd
import pytest
import yaml

from fonpr.sim.milan import aggregate_activity, convert_milan, read_milan_files, select_squares
from fonpr.sim.trace import load_trace

BASE_MS = 1_383_260_400_000  # 2013-11-01 00:00 UTC-ish, matches source era


def write_fixture(path, hours=3, squares=(1, 2), gap_intervals=()):
    """Milan-format TSV: two country-code rows per (square, interval)."""
    rows = []
    n = hours * 6  # 10-minute intervals
    for i in range(n):
        if i in gap_intervals:
            continue
        ts = BASE_MS + i * 600_000
        for sq in squares:
            # Diurnal-ish ramp, distinct per square; split across two
            # country codes to exercise the aggregation.
            activity = 100.0 * sq + 10.0 * i
            rows.append(f"{sq}\t{ts}\t39\t1\t1\t1\t1\t{activity * 0.7}")
            rows.append(f"{sq}\t{ts}\t0\t\t\t\t\t{activity * 0.3}")
    path.write_text("\n".join(rows) + "\n")
    return path


def test_read_and_aggregate_sums_squares_and_country_codes(tmp_path):
    fixture = write_fixture(tmp_path / "milan.txt")
    raw = read_milan_files([fixture])
    activity = aggregate_activity(select_squares(raw, None, None))
    # Interval 0: squares 1+2 -> (100 + 200) * (0.7 + 0.3) = 300.
    assert activity.iloc[0] == pytest.approx(300.0)
    assert len(activity) == 18


def test_square_selection(tmp_path):
    fixture = write_fixture(tmp_path / "milan.txt")
    raw = read_milan_files([fixture])
    only_two = aggregate_activity(select_squares(raw, [2], None))
    assert only_two.iloc[0] == pytest.approx(200.0)
    top_one = aggregate_activity(select_squares(raw, None, 1))
    assert top_one.iloc[0] == pytest.approx(200.0)  # square 2 is busiest
    with pytest.raises(ValueError, match="not present"):
        select_squares(raw, [99], None)
    with pytest.raises(ValueError, match="not both"):
        select_squares(raw, [1], 1)


def test_small_gap_interpolated_large_gap_rejected(tmp_path):
    ok = write_fixture(tmp_path / "ok.txt", gap_intervals=(5,))
    activity = aggregate_activity(read_milan_files([ok]))
    assert not activity.isna().any()
    bad = write_fixture(tmp_path / "bad.txt", gap_intervals=tuple(range(4, 11)))
    with pytest.raises(ValueError, match="refusing to fabricate"):
        aggregate_activity(read_milan_files([bad]))


def test_convert_produces_loadable_trace_with_scaled_peak(tmp_path):
    fixture = write_fixture(tmp_path / "milan.txt")
    out = tmp_path / "milan.parquet"
    outputs = convert_milan([fixture], out, peak_bytes_per_sec=2.0e7)
    trace = load_trace(outputs[0])
    assert trace.throughput_bytes_per_sec.max() == pytest.approx(2.0e7)
    # 3 hours at 10-min source -> tick grid at 1 min.
    assert len(trace) == 17 * 10 + 1
    meta = yaml.safe_load((tmp_path / "milan.meta.yaml").read_text())
    assert "ODbL" in meta["attribution"]
    assert meta["scale_activity_to_bytes_per_sec"] > 0


def test_split_holds_out_final_days(tmp_path):
    fixture = write_fixture(tmp_path / "milan.txt", hours=72)
    out = tmp_path / "milan.parquet"
    outputs = convert_milan([fixture], out, split_eval_days=1)
    assert [p.name for p in outputs] == ["milan-train.parquet", "milan-eval.parquet"]
    train, eval_ = (load_trace(p) for p in outputs)
    assert train.timestamp.max() < eval_.timestamp.min()
    span = eval_.timestamp.max() - eval_.timestamp.min()
    assert span <= pd.Timedelta(days=1)
    # No overlap: every tick lands in exactly one split.
    # 72h of 10-min intervals -> (432-1)*10 + 1 ticks at 1-min granularity.
    assert len(train) + len(eval_) == 431 * 10 + 1


def test_split_rejecting_empty_side(tmp_path):
    fixture = write_fixture(tmp_path / "milan.txt", hours=3)
    with pytest.raises(ValueError, match="empty split"):
        convert_milan([fixture], tmp_path / "m.parquet", split_eval_days=5)


def test_replay_through_env(tmp_path):
    """End-to-end: Milan fixture -> trace -> sim episode."""
    from fonpr.sim import FONPRSimEnv, SimConfig
    from fonpr.sim.config import TimeConfig, TrafficConfig

    fixture = write_fixture(tmp_path / "milan.txt", hours=30)
    outputs = convert_milan([fixture], tmp_path / "milan.parquet")
    cfg = SimConfig(
        time=TimeConfig(episode_days=1.0),
        traffic=TrafficConfig(trace_path=str(outputs[0])),
    )
    env = FONPRSimEnv(cfg)
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    assert info["offered_load"] > 0
