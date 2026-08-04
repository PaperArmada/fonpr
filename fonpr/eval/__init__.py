"""
Evaluation harness (S4): the arbiter of every performance claim.
"""

from fonpr.eval.harness import (
    EvalConfig,
    PerturbConfig,
    evaluate,
    perturbed_world,
    run_eval_cli,
    scenario_config,
)

__all__ = [
    "EvalConfig",
    "PerturbConfig",
    "evaluate",
    "perturbed_world",
    "run_eval_cli",
    "scenario_config",
]
