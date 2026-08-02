"""
Policy seam (S3): baselines, oracle, and learned-agent wrappers.
"""

from fonpr.policies.base import Policy, current_instance, observed_throughput
from fonpr.policies.baselines import (
    ForecastPolicy,
    NoopPolicy,
    ReactivePolicy,
    ThresholdPolicy,
    make_baselines,
)
from fonpr.policies.oracle import OraclePolicy, plan_oracle_actions

__all__ = [
    "ForecastPolicy",
    "NoopPolicy",
    "OraclePolicy",
    "Policy",
    "ReactivePolicy",
    "ThresholdPolicy",
    "current_instance",
    "make_baselines",
    "observed_throughput",
    "plan_oracle_actions",
]
