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

"""Metrics helpers for Hermes telemetry."""

from __future__ import annotations

from typing import Any

from opentelemetry import metrics as metrics_api

from .version import __version__


class HermesMetrics:
    def __init__(self, meter_provider=None):
        meter = metrics_api.get_meter(
            __name__,
            __version__,
            meter_provider=meter_provider,
        )
        self._calls_count = meter.create_counter(
            name="genai_calls_count",
            description="GenAI call count",
            unit="1",
        )
        self._calls_error_count = meter.create_counter(
            name="genai_calls_error_count",
            description="GenAI call error count",
            unit="1",
        )
        self._calls_duration_seconds = meter.create_histogram(
            name="genai_calls_duration_seconds",
            description="GenAI call duration",
            unit="s",
        )
        self._llm_usage_tokens = meter.create_counter(
            name="genai_llm_usage_tokens",
            description="LLM token usage",
            unit="1",
        )

    @staticmethod
    def _attrs(
        provider: str, model: str, operation: str = "chat"
    ) -> dict[str, Any]:
        return {
            "callType": "gen_ai",
            "callKind": "internal",
            "rpcType": 2100,
            "modelName": model,
            "provider": provider,
            "spanKind": "LLM",
            "rpc": f"{operation} {model}",
        }

    def record_llm_call(
        self, provider: str, model: str, operation: str = "chat"
    ):
        self._calls_count.add(1, self._attrs(provider, model, operation))

    def record_llm_error(
        self, provider: str, model: str, operation: str = "chat"
    ):
        self._calls_error_count.add(1, self._attrs(provider, model, operation))

    def record_llm_duration(
        self,
        provider: str,
        model: str,
        duration_seconds: float,
        operation: str = "chat",
    ):
        self._calls_duration_seconds.record(
            duration_seconds,
            self._attrs(provider, model, operation),
        )

    def record_llm_tokens(
        self,
        provider: str,
        model: str,
        token_type: str,
        value: int,
        operation: str = "chat",
    ):
        if value <= 0:
            return
        attrs = self._attrs(provider, model, operation)
        attrs["tokenType"] = token_type
        self._llm_usage_tokens.add(value, attrs)
