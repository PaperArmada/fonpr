"""
Single command-line entry point for FONPR.

Every user-facing operation is a subcommand registered here; no other
module defines an entry point.
"""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fonpr",
        description="RL/control agents for cost-optimal operation of a cloud-native 5G core.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level name.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser(
        "eval", help="Run the evaluation harness (S4) over policies and scenarios."
    )
    eval_parser.add_argument("--config", default=None, help="Path to an eval config YAML.")
    eval_parser.add_argument("--out", default="results", help="Output directory root.")
    eval_parser.add_argument(
        "--quick", action="store_true", help="Reduced seeds/episode length for smoke runs."
    )
    eval_parser.add_argument(
        "--dqn-checkpoint",
        action="append",
        default=None,
        metavar="[LABEL=]PATH",
        help="Trained DQN checkpoint to include (repeatable). Optional LABEL= "
        "prefix names the policy row; default label is 'dqn'.",
    )
    eval_parser.add_argument(
        "--time-features",
        action="store_true",
        help="Evaluate on the enriched sin/cos time-of-day observation (ADR-0002).",
    )
    eval_parser.add_argument(
        "--energy",
        action="store_true",
        help="Cost infrastructure via the S13 power model instead of the price table.",
    )
    eval_parser.add_argument(
        "--pool",
        action="store_true",
        help="Run the replica-count pool plant (S14, ADR-0004) instead of binary sizing.",
    )

    train_parser = subparsers.add_parser(
        "train", help="Train the DQN agent in the simulator (S5; requires the [rl] extra)."
    )
    train_parser.add_argument("--config", default=None, help="Path to a sim config YAML.")
    train_parser.add_argument("--out", default="runs", help="Output directory root.")
    train_parser.add_argument("--steps", type=int, default=500_000, help="Training steps.")
    train_parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    train_parser.add_argument(
        "--algo",
        default="ppo",  # ADR-0003
        choices=["dqn", "ppo"],
        help="Training algorithm (default ppo per ADR-0003; dqn retained for comparison).",
    )

    agent_parser = subparsers.add_parser(
        "run-agent", help="Run the live control loop (S12); dry-run actuation by default."
    )
    agent_parser.add_argument("--config", default=None, help="Path to a loop config YAML.")
    agent_parser.add_argument(
        "--once", action="store_true", help="Run a single iteration and exit."
    )

    trace_parser = subparsers.add_parser("trace", help="Trace utilities (S1.6).")
    trace_sub = trace_parser.add_subparsers(dest="trace_command", required=True)
    pull_parser = trace_sub.add_parser(
        "pull", help="Export a UPF throughput trace from a live Prometheus."
    )
    pull_parser.add_argument("--endpoint", required=True, help="Prometheus ip:port.")
    pull_parser.add_argument("--hours", type=float, default=24.0, help="Lookback window.")
    pull_parser.add_argument("--out", default="trace.parquet", help="Output file path.")
    milan_parser = trace_sub.add_parser(
        "convert-milan",
        help="Convert Telecom Italia Milan dataset files to the trace schema.",
    )
    milan_parser.add_argument("inputs", nargs="+", help="Milan telecom TSV file path(s).")
    milan_parser.add_argument("--out", default="milan.parquet", help="Output trace path.")
    milan_parser.add_argument(
        "--squares", default=None, help="Comma-separated square ids to aggregate."
    )
    milan_parser.add_argument(
        "--top", type=int, default=None, help="Aggregate the N busiest squares."
    )
    milan_parser.add_argument(
        "--peak-bytes-per-sec",
        type=float,
        default=2.0e7,
        help="Scale the series so its peak equals this load (default matches D7 calibration).",
    )
    milan_parser.add_argument(
        "--split-eval-days",
        type=int,
        default=0,
        help="Write separate -train/-eval files, holding out the last N days.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "eval":
        from fonpr.eval.harness import run_eval_cli

        return run_eval_cli(
            config_path=args.config,
            out_root=args.out,
            quick=args.quick,
            dqn_checkpoint=args.dqn_checkpoint,
            time_features=args.time_features,
            energy=args.energy,
            pool=args.pool,
        )
    if args.command == "train":
        from fonpr.train import run_train_cli

        return run_train_cli(
            config_path=args.config,
            out_root=args.out,
            steps=args.steps,
            seed=args.seed,
            algorithm=args.algo,
        )
    if args.command == "run-agent":
        from fonpr.loop import run_agent_cli

        return run_agent_cli(config_path=args.config, once=args.once)
    if args.command == "trace":
        if args.trace_command == "pull":
            from fonpr.sim.trace import pull_trace

            out = pull_trace(endpoint=args.endpoint, hours=args.hours, out_path=args.out)
            logger.info("trace written to %s", out)
            return 0
        if args.trace_command == "convert-milan":
            from pathlib import Path

            from fonpr.sim.milan import convert_milan

            squares = (
                [int(s) for s in args.squares.split(",")] if args.squares else None
            )
            outputs = convert_milan(
                input_paths=[Path(p) for p in args.inputs],
                out_path=Path(args.out),
                squares=squares,
                top=args.top,
                peak_bytes_per_sec=args.peak_bytes_per_sec,
                split_eval_days=args.split_eval_days,
            )
            logger.info("trace(s) written: %s", [str(p) for p in outputs])
            return 0
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
