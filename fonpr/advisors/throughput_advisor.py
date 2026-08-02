"""
Typed live advisor (S12): produces observations with the exact shape and
semantics of the simulator's (S1.1), so any Policy deploys unchanged.

Requires the [live] extra at runtime; tests inject a stub client.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)

# Served user-plane throughput, bytes/sec, summed over UPF pods.
THROUGHPUT_QUERY = (
    "sum(rate(container_network_transmit_bytes_total{pod=~'open5gs-upf.*'}[2m]))"
)
# Instance type label of the node currently hosting the UPF.
UPF_NODE_QUERY = (
    "kube_node_labels * on(node) group_right() "
    "count by (node) (kube_pod_info{pod=~'open5gs-upf.*'})"
)
INSTANCE_LABEL = "label_node_kubernetes_io_instance_type"


class ThroughputAdvisor:
    """Builds (samples, 3) observations from a live Prometheus."""

    def __init__(
        self,
        endpoint: str,
        large_instance_type: str,
        small_instance_type: str,
        client=None,
    ):
        self._endpoint = endpoint
        self._large = large_instance_type
        self._small = small_instance_type
        self._client = client  # tests inject; live path connects lazily

    def _prom(self):
        if self._client is None:
            from prometheus_api_client import PrometheusConnect  # [live] extra

            self._client = PrometheusConnect(
                url=f"http://{self._endpoint}", disable_ssl=True
            )
        return self._client

    def observe(self, window_minutes: int, sample_rate_per_minute: int = 1) -> np.ndarray:
        """One observation window ending now; raises on empty query results."""
        prom = self._prom()
        end = datetime.now(UTC)
        start = end - timedelta(minutes=window_minutes)
        step_seconds = int(60 / sample_rate_per_minute)

        series = prom.custom_query_range(
            THROUGHPUT_QUERY, start_time=start, end_time=end, step=str(step_seconds)
        )
        if not series:
            raise RuntimeError(f"no UPF throughput series from {self._endpoint}")
        values = {float(t): float(v) for t, v in series[0]["values"]}
        n = window_minutes * sample_rate_per_minute
        grid = [start.timestamp() + i * step_seconds for i in range(1, n + 1)]
        timestamps = sorted(values)
        throughput = np.interp(grid, timestamps, [values[t] for t in timestamps])

        node_rows = prom.custom_query(UPF_NODE_QUERY)
        if not node_rows:
            raise RuntimeError(f"no UPF node-label series from {self._endpoint}")
        instance_type = node_rows[0]["metric"].get(INSTANCE_LABEL, "")
        large_on = 1.0 if instance_type == self._large else 0.0
        small_on = 1.0 if instance_type == self._small else 0.0
        if not (large_on or small_on):
            logger.warning("UPF node instance type %r matches neither tier", instance_type)

        obs = np.stack(
            [throughput, np.full(n, large_on), np.full(n, small_on)], axis=1
        ).astype(np.float32)
        return obs
