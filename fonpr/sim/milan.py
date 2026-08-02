"""
Milan (Telecom Italia Big Data Challenge) trace converter (S1.6 source).

Converts the open Milan telecom activity dataset (Barlacchi et al. 2015,
Harvard Dataverse doi:10.7910/DVN/EGZHFV, ODbL) into the FONPR trace
schema. The source records per-square, per-country-code activity at
10-minute intervals; we aggregate a square selection, keep the *shape*,
and scale its peak to the plant's capacity calibration — the source's
"internet activity" unit is a proxy, not bytes/sec, and is treated as such.

Source TSV columns (tab-separated, no header):
    square_id, time_interval(epoch ms), country_code,
    sms_in, sms_out, call_in, call_out, internet_traffic
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

SOURCE_INTERVAL_MINUTES = 10
# Source gaps longer than this are an error: interpolating through an
# outage would fabricate demand.
MAX_SOURCE_GAP_INTERVALS = 6  # one hour

COLUMNS = [
    "square_id",
    "time_interval",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet_traffic",
]

ATTRIBUTION = (
    "Derived from 'Telecommunications - SMS, Call, Internet' (Telecom Italia "
    "Big Data Challenge), Barlacchi et al. 2015, doi:10.7910/DVN/EGZHFV, "
    "licensed ODbL 1.0. This derived series carries the same license."
)


def read_milan_files(paths: list[Path]) -> pd.DataFrame:
    """Read raw Milan TSV files down to (square_id, time_interval, internet)."""
    frames = []
    for path in paths:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=COLUMNS,
            usecols=["square_id", "time_interval", "internet_traffic"],
        )
        frames.append(df.dropna(subset=["internet_traffic"]))
    if not frames:
        raise ValueError("no input files")
    return pd.concat(frames, ignore_index=True)


def select_squares(raw: pd.DataFrame, squares: list[int] | None, top: int | None) -> pd.DataFrame:
    """Filter to an explicit square list, the top-N busiest squares, or all."""
    if squares and top:
        raise ValueError("pass either an explicit square list or --top, not both")
    if squares:
        missing = set(squares) - set(raw.square_id.unique())
        if missing:
            raise ValueError(f"square id(s) not present in input: {sorted(missing)}")
        return raw[raw.square_id.isin(squares)]
    if top:
        busiest = (
            raw.groupby("square_id").internet_traffic.sum().nlargest(top).index
        )
        return raw[raw.square_id.isin(busiest)]
    return raw


def aggregate_activity(selected: pd.DataFrame) -> pd.Series:
    """Sum internet activity over squares and country codes per interval.

    Returns a Series indexed by UTC timestamp on the 10-minute source grid,
    with gaps up to MAX_SOURCE_GAP_INTERVALS linearly interpolated and
    longer gaps rejected.
    """
    series = selected.groupby("time_interval").internet_traffic.sum().sort_index()
    if len(series) < 2:
        raise ValueError("need at least two source intervals")
    index = pd.to_datetime(series.index, unit="ms", utc=True)
    series.index = index

    full_grid = pd.date_range(
        index[0], index[-1], freq=f"{SOURCE_INTERVAL_MINUTES}min", tz="UTC"
    )
    on_grid = series.reindex(full_grid)
    gap_run = on_grid.isna().astype(int).groupby(on_grid.notna().cumsum()).sum()
    worst_gap = int(gap_run.max()) if len(gap_run) else 0
    if worst_gap > MAX_SOURCE_GAP_INTERVALS:
        raise ValueError(
            f"source gap of {worst_gap} intervals exceeds "
            f"{MAX_SOURCE_GAP_INTERVALS}; refusing to fabricate demand"
        )
    return on_grid.interpolate(method="linear", limit_area="inside")


def convert_milan(
    input_paths: list[Path],
    out_path: Path,
    squares: list[int] | None = None,
    top: int | None = None,
    peak_bytes_per_sec: float = 2.0e7,
    tick_minutes: float = 1.0,
    split_eval_days: int = 0,
) -> list[Path]:
    """Full pipeline; returns the written trace file path(s)."""
    raw = read_milan_files(input_paths)
    activity = aggregate_activity(select_squares(raw, squares, top))

    scale = peak_bytes_per_sec / float(activity.max())
    load = activity * scale

    # Interpolate onto the tick grid the sim consumes (loader's gap rule
    # requires trace sampling finer than 3 ticks).
    n_ticks = int((load.index[-1] - load.index[0]).total_seconds() // (tick_minutes * 60)) + 1
    grid = load.index[0] + pd.to_timedelta(np.arange(n_ticks) * tick_minutes, unit="m")
    ticks = np.interp(
        grid.astype("int64").to_numpy(),
        load.index.astype("int64").to_numpy(),
        load.to_numpy(dtype=np.float64),
    )
    frame = pd.DataFrame({"timestamp": grid, "throughput_bytes_per_sec": ticks})

    out_path = Path(out_path)
    outputs: list[Path] = []
    if split_eval_days > 0:
        cutoff = grid[-1] - pd.Timedelta(days=split_eval_days)
        train = frame[frame.timestamp <= cutoff]
        eval_ = frame[frame.timestamp > cutoff]
        if train.empty or eval_.empty:
            raise ValueError("split_eval_days leaves an empty split")
        for part, name in ((train, "train"), (eval_, "eval")):
            path = out_path.with_name(f"{out_path.stem}-{name}{out_path.suffix}")
            part.to_parquet(path, index=False)
            outputs.append(path)
    else:
        frame.to_parquet(out_path, index=False)
        outputs.append(out_path)

    meta = {
        "attribution": ATTRIBUTION,
        "source_files": [str(p) for p in input_paths],
        "squares": squares or (f"top-{top}" if top else "all"),
        "scale_activity_to_bytes_per_sec": scale,
        "peak_bytes_per_sec": peak_bytes_per_sec,
        "tick_minutes": tick_minutes,
        "split_eval_days": split_eval_days,
        "outputs": [str(p) for p in outputs],
    }
    meta_path = out_path.with_suffix(".meta.yaml")
    with open(meta_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    logger.info("wrote %s (+ %s)", [str(p) for p in outputs], meta_path.name)
    return outputs
