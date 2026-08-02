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

    train_parser = subparsers.add_parser(
        "train", help="Train the DQN agent in the simulator (S5; requires the [rl] extra)."
    )
    train_parser.add_argument("--config", default=None, help="Path to a sim config YAML.")
    train_parser.add_argument("--out", default="runs", help="Output directory root.")
    train_parser.add_argument("--steps", type=int, default=500_000, help="Training steps.")
    train_parser.add_argument("--seed", type=int, default=0, help="Training seed.")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "eval":
        from fonpr.eval.harness import run_eval_cli

        return run_eval_cli(config_path=args.config, out_root=args.out, quick=args.quick)
    if args.command == "train":
        from fonpr.train import run_train_cli

        return run_train_cli(
            config_path=args.config, out_root=args.out, steps=args.steps, seed=args.seed
        )
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
