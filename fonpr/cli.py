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

    train_parser = subparsers.add_parser(
        "train", help="Train the DQN agent in the simulator (S5; requires the [rl] extra)."
    )
    train_parser.add_argument("--config", default=None, help="Path to a sim config YAML.")
    train_parser.add_argument("--out", default="runs", help="Output directory root.")
    train_parser.add_argument("--steps", type=int, default=500_000, help="Training steps.")
    train_parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    train_parser.add_argument(
        "--algo", default="dqn", choices=["dqn", "ppo"], help="Training algorithm."
    )

    trace_parser = subparsers.add_parser("trace", help="Trace utilities (S1.6).")
    trace_sub = trace_parser.add_subparsers(dest="trace_command", required=True)
    pull_parser = trace_sub.add_parser(
        "pull", help="Export a UPF throughput trace from a live Prometheus."
    )
    pull_parser.add_argument("--endpoint", required=True, help="Prometheus ip:port.")
    pull_parser.add_argument("--hours", type=float, default=24.0, help="Lookback window.")
    pull_parser.add_argument("--out", default="trace.parquet", help="Output file path.")

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
    if args.command == "trace":
        from fonpr.sim.trace import pull_trace

        out = pull_trace(endpoint=args.endpoint, hours=args.hours, out_path=args.out)
        logger.info("trace written to %s", out)
        return 0
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
