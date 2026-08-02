"""
Trace-replay offered load (S1.6): drive the simulator with recorded
throughput instead of synthesis.

File contract: Parquet (or CSV import) with columns
``timestamp`` (UTC datetimes) and ``throughput_bytes_per_sec`` (float).
Samples are linearly interpolated onto the sim tick grid; a gap longer
than 3x the tick is an error, never silently interpolated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fonpr.sim.config import TimeConfig

REQUIRED_COLUMNS = ("timestamp", "throughput_bytes_per_sec")
MAX_GAP_TICKS = 3


def load_trace(path: str | Path) -> pd.DataFrame:
    """Read a trace file (Parquet, or CSV import) and validate its schema."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        df = pd.read_parquet(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: trace is missing column(s) {sorted(missing)}")
    if len(df) < 2:
        raise ValueError(f"{path}: trace needs at least two samples")
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Normalize to ns precision: Parquet round-trips as us, CSV parses as ns.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    if (df.throughput_bytes_per_sec < 0).any():
        raise ValueError(f"{path}: negative throughput values")
    return df


def resample_to_ticks(trace: pd.DataFrame, time_cfg: TimeConfig) -> np.ndarray:
    """Linear-interpolate the trace onto the sim tick grid (S1.6).

    Gaps longer than ``MAX_GAP_TICKS`` ticks are an error.
    """
    timestamps = trace.timestamp.astype("int64").to_numpy() / 1e9  # seconds
    values = trace.throughput_bytes_per_sec.to_numpy(dtype=np.float64)

    tick_seconds = time_cfg.tick_minutes * 60.0
    gaps = np.diff(timestamps)
    worst = gaps.max()
    if worst > MAX_GAP_TICKS * tick_seconds:
        raise ValueError(
            f"trace gap of {worst:.0f}s exceeds {MAX_GAP_TICKS} ticks "
            f"({MAX_GAP_TICKS * tick_seconds:.0f}s); fix the trace, do not interpolate"
        )

    span = timestamps[-1] - timestamps[0]
    n_ticks = int(span // tick_seconds) + 1
    grid = timestamps[0] + np.arange(n_ticks) * tick_seconds
    return np.interp(grid, timestamps, values)


def pull_trace(
    endpoint: str, hours: float, out_path: str | Path, step_seconds: int = 60
) -> Path:
    """Export a UPF-throughput trace from a live Prometheus into the trace
    schema (S1.6). Requires the [live] extra; [trace] extra for Parquet."""
    import datetime

    from prometheus_api_client import PrometheusConnect  # [live] extra

    prom = PrometheusConnect(url=f"http://{endpoint}", disable_ssl=True)
    end = datetime.datetime.now(datetime.UTC)
    start = end - datetime.timedelta(hours=hours)
    query = "sum(rate(container_network_transmit_bytes_total{pod=~'open5gs-upf.*'}[2m]))"
    raw = prom.custom_query_range(query, start_time=start, end_time=end, step=str(step_seconds))
    if not raw:
        raise RuntimeError(f"no UPF throughput series returned from {endpoint}")
    values = raw[0]["values"]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([float(t) for t, _ in values], unit="s", utc=True),
            "throughput_bytes_per_sec": [float(v) for _, v in values],
        }
    )
    out_path = Path(out_path)
    if out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)
    return out_path


class ReplayTrafficModel:
    """Drop-in for TrafficModel: serves a recorded trace tick by tick.

    Raises when the episode consumes more ticks than the trace holds —
    explicit failure beats silently looping the tape.
    """

    def __init__(self, ticks: np.ndarray):
        if len(ticks) == 0:
            raise ValueError("empty trace")
        self._ticks = np.asarray(ticks, dtype=np.float64)
        self.reset()

    @classmethod
    def from_file(cls, path: str | Path, time_cfg: TimeConfig) -> ReplayTrafficModel:
        return cls(resample_to_ticks(load_trace(path), time_cfg))

    def reset(self) -> None:
        self._cursor = 0

    def advance(self, n_ticks: int) -> np.ndarray:
        end = self._cursor + n_ticks
        if end > len(self._ticks):
            raise RuntimeError(
                f"trace exhausted: needed tick {end}, trace has {len(self._ticks)} "
                "(shorten episode_days or record a longer trace)"
            )
        out = self._ticks[self._cursor : end].copy()
        self._cursor = end
        return out
