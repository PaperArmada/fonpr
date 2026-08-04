"""
Result-bundle writer (S4.3): results.csv, results.md, plots, run metadata.

Generated artifacts are the only home of performance numbers; nothing here
is ever hand-edited. Colors follow the entity (policy), never its rank, and
the oracle renders as a neutral reference bound, not a competing series.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import platform
from importlib import metadata
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Fixed categorical assignment (validated palette; see docs/SPECS.md S4.3).
POLICY_COLORS = {
    "noop": "#2a78d6",
    "threshold": "#eb6834",
    "reactive": "#1baf7a",
    "forecast": "#eda100",
    "mpc": "#12999e",
    "dqn": "#e87ba4",
}
ORACLE_COLOR = "#6f6e69"  # neutral: a bound, not a contender
OFFERED_COLOR = "#b9b7ae"
CAPACITY_COLOR = "#3d3d3a"

METRIC_LABELS = {
    "total_cost_usd": "Total operating cost ($)",
    "violation_minutes": "SLO violation minutes",
    "action_churn": "Actions taken",
    "regret_usd": "Regret vs oracle ($)",
}
# Summarized alongside METRIC_LABELS; shown in results.md only for
# energy-variant runs (S13), where it is nonzero.
ENERGY_METRIC = "energy_kwh"


def _ci95(series: pd.Series) -> float:
    n = len(series)
    return 1.96 * series.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% CI per scenario (x robustness cell) x policy."""
    keys = ["scenario", "perturb", "policy"] if "perturb" in results else ["scenario", "policy"]
    grouped = results.groupby(keys)
    parts = {}
    for metric in (*METRIC_LABELS, ENERGY_METRIC):
        parts[f"{metric}_mean"] = grouped[metric].mean()
        parts[f"{metric}_ci95"] = grouped[metric].apply(_ci95)
    return pd.DataFrame(parts).reset_index()


def _sections(summary: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """(display key, heading, rows) per scenario (x cell), in frame order."""
    out = []
    if "perturb" in summary:
        pairs = summary[["scenario", "perturb"]].drop_duplicates().itertuples(index=False)
        for scenario, perturb in pairs:
            rows = summary[(summary.scenario == scenario) & (summary.perturb == perturb)]
            display = scenario if not perturb else f"{scenario}__{perturb}"
            heading = scenario if not perturb else f"{scenario} · cell: {perturb}"
            out.append((display, heading, rows))
    else:
        for scenario in summary.scenario.unique():
            rows = summary[summary.scenario == scenario]
            out.append((scenario, scenario, rows))
    return out


def _policy_order(policies: list[str]) -> list[str]:
    known = [p for p in POLICY_COLORS if p in policies]
    rest = sorted(p for p in policies if p not in POLICY_COLORS and p != "oracle")
    return known + rest + (["oracle"] if "oracle" in policies else [])


def plot_cost_by_policy(data: pd.DataFrame, scenario: str, path: Path) -> None:
    """``data``: one section's summary rows; ``scenario``: its display key."""
    order = [p for p in _policy_order(list(data.policy)) if p != "oracle"]
    rows = data.set_index("policy").loc[order]
    oracle = data[data.policy == "oracle"]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [POLICY_COLORS.get(p, "#4a3aa7") for p in order]
    bars = ax.bar(
        order,
        rows["total_cost_usd_mean"],
        yerr=rows["total_cost_usd_ci95"],
        color=colors,
        width=0.6,
        capsize=4,
        error_kw={"ecolor": "#3d3d3a", "elinewidth": 1},
    )
    for bar, value in zip(bars, rows["total_cost_usd_mean"], strict=True):
        ax.annotate(
            f"${value:,.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#3d3d3a",
            xytext=(0, 3),
            textcoords="offset points",
        )
    if not oracle.empty:
        bound = float(oracle["total_cost_usd_mean"].iloc[0])
        ax.axhline(bound, color=ORACLE_COLOR, linestyle="--", linewidth=1.5)
        # Fixed top-corner placement: never collides with bar value labels.
        ax.annotate(
            f"-- oracle bound: ${bound:,.2f}",
            (0.99, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            color=ORACLE_COLOR,
        )
    ax.set_ylabel("Total operating cost ($, lower is better)")
    ax.set_title(f"Cost by policy — {scenario} (mean ± 95% CI)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timelines(
    timelines: dict[str, dict[str, np.ndarray]],
    scenario: str,
    tick_minutes: float,
    path: Path,
) -> None:
    """Small multiples: offered vs served vs capacity per policy, one seed."""
    policies = _policy_order(list(timelines))
    if not policies:
        return
    fig, axes = plt.subplots(
        len(policies), 1, figsize=(9, 1.9 * len(policies)), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, policy in zip(axes, policies, strict=True):
        series = timelines[policy]
        days = np.arange(len(series["offered"])) * tick_minutes / (24 * 60)
        color = ORACLE_COLOR if policy == "oracle" else POLICY_COLORS.get(policy, "#4a3aa7")
        ax.plot(days, series["offered"] / 1e6, color=OFFERED_COLOR, linewidth=1, label="offered")
        ax.plot(days, series["served"] / 1e6, color=color, linewidth=1.2, label="served")
        ax.plot(
            days,
            series["capacity"] / 1e6,
            color=CAPACITY_COLOR,
            linewidth=1.2,
            linestyle="--",
            label="capacity",
        )
        ax.set_ylabel("MB/s")
        ax.set_title(policy, loc="left", fontsize=10)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
    axes[-1].set_xlabel("Days")
    fig.suptitle(f"Load vs capacity — {scenario} (one evaluation seed)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, results: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Evaluation results",
        "",
        "_Generated by `fonpr eval`; do not hand-edit (CLAUDE.md)._",
        "",
    ]
    show_energy = bool((results[ENERGY_METRIC].abs() > 0).any())
    for _display, heading, data in _sections(summary):
        order = _policy_order(list(data.policy))
        lines += [f"## {heading}", ""]
        header = "| Policy | Total cost ($) | SLO violation (min) | Actions | Regret ($) |"
        rule = "|---|---|---|---|---|"
        if show_energy:
            header += " Energy (kWh) |"
            rule += "---|"
        lines += [header, rule]
        for policy in order:
            row = data[data.policy == policy].iloc[0]
            cells = (
                f"| {policy} "
                f"| {row.total_cost_usd_mean:,.2f} ± {row.total_cost_usd_ci95:,.2f} "
                f"| {row.violation_minutes_mean:,.0f} ± {row.violation_minutes_ci95:,.0f} "
                f"| {row.action_churn_mean:,.1f} "
                f"| {row.regret_usd_mean:,.2f} |"
            )
            if show_energy:
                cells += f" {row.energy_kwh_mean:,.2f} ± {row.energy_kwh_ci95:,.2f} |"
            lines.append(cells)
        lines.append("")
    n_seeds = results.seed.nunique()
    lines += [
        f"Seeds per scenario: {n_seeds}. CI: normal-approximation 95%.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# Distributions whose versions shape numeric results; absent ones are omitted.
_ENVIRONMENT_DISTS = (
    "numpy",
    "pandas",
    "gymnasium",
    "PyYAML",
    "matplotlib",
    "stable-baselines3",
    "torch",
    "pyarrow",
)


def environment_versions() -> dict:
    """Python and key package versions: the environment leg of reproducibility.

    Seed + config + git SHA pin the code and inputs; this pins the third
    input, the library stack, so bundles produced on different machines are
    comparable (S4.3).
    """
    packages = {}
    for dist in _ENVIRONMENT_DISTS:
        try:
            packages[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return {"python": platform.python_version(), "packages": packages}


def write_report(results, timelines, eval_cfg, out_root: Path) -> Path:
    from fonpr.eval.harness import git_sha

    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = out_root / f"run-{stamp}"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_dir / "results.csv", index=False)
    summary = summarize(results)
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_markdown(summary, results, out_dir / "results.md")

    from fonpr.eval.harness import scenario_config

    for display, _heading, data in _sections(summary):
        plot_cost_by_policy(data, display, plots_dir / f"cost_{display}.png")
        scenario = display.split("__", 1)[0]
        tick = scenario_config(scenario, eval_cfg.episode_days).time.tick_minutes
        plot_timelines(
            timelines.get(display, {}), display, tick, plots_dir / f"timeline_{display}.png"
        )

    meta = {
        "git_sha": git_sha(),
        "generated_utc": stamp,
        "eval_config": dataclasses.asdict(eval_cfg),
        "environment": environment_versions(),
    }
    with open(out_dir / "run_meta.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    logger.info("report bundle: %s", out_dir)
    return out_dir
