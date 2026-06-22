# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent-Reach specific metrics.

Custom metrics derived from Agent-Reach Channel/Probe invocations. Per the
ARMS GenAI metrics semantic conventions, generic request/error/duration
metrics are derived from Span data by the backend and are NOT recorded here.
Only Agent-Reach-specific diagnostics (channel health, probe outcome) are
recorded as custom gauges/counters/histograms.
"""

from __future__ import annotations

import logging
from typing import Optional

from opentelemetry.metrics import Meter

logger = logging.getLogger(__name__)

_METER_NAME = "loongsuite-instrumentation-agent-reach"


class AgentReachMetrics:
    """Holds Agent-Reach specific metric instruments.

    All instruments are created lazily; a missing meter (None) is tolerated
    so unit tests that don't configure a MeterProvider can still run.
    """

    def __init__(self, meter: Optional[Meter]) -> None:
        self._meter = meter
        self._channel_status = None
        self._probe_counter = None
        self._probe_duration = None
        if meter is None:
            return
        try:
            self._channel_status = meter.create_gauge(
                name="agent_reach_channel_status",
                description="Channel health status (1=ok, 0=otherwise)",
                unit="1",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to create channel_status gauge: %s", e)
        try:
            self._probe_counter = meter.create_counter(
                name="agent_reach_probe_total",
                description="Number of probe_command executions",
                unit="1",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to create probe_total counter: %s", e)
        try:
            self._probe_duration = meter.create_histogram(
                name="agent_reach_probe_duration_seconds",
                description="probe_command execution duration in seconds",
                unit="s",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to create probe_duration histogram: %s", e)

    def record_channel_status(self, channel: str, status: str) -> None:
        if self._channel_status is None:
            return
        try:
            self._channel_status.set(
                1 if status == "ok" else 0,
                attributes={"channel": channel, "status": status},
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to record channel_status: %s", e)

    def record_probe(
        self, cmd: str, status: str, duration_seconds: float
    ) -> None:
        attrs = {"cmd": cmd, "status": status}
        if self._probe_counter is not None:
            try:
                self._probe_counter.add(1, attributes=attrs)
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to record probe_total: %s", e)
        if self._probe_duration is not None:
            try:
                self._probe_duration.record(
                    max(0.0, float(duration_seconds)),
                    attributes={"cmd": cmd},
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to record probe_duration: %s", e)


__all__: list = ["AgentReachMetrics"]
