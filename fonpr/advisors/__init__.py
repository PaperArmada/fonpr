"""
Advisor seam: ingest live telemetry and shape it for policies.

``PromClient`` (legacy) needs the [live] extra; it is re-exported lazily so
importing this package — e.g. for ThroughputAdvisor with an injected test
client — works without prometheus_api_client installed.
"""


def __getattr__(name):
    if name == "PromClient":
        from .prometheus_client_advisor import PromClient

        return PromClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PromClient"]
