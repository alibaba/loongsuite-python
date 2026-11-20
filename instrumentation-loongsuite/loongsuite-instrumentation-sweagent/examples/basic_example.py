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

"""Minimal example: instrument run hooks and export spans to the console."""

from __future__ import annotations

from unittest.mock import MagicMock

from sweagent.run.hooks.abstract import CombinedRunHooks
from sweagent.types import AgentInfo, AgentRunResult

from opentelemetry.instrumentation.sweagent import SweagentInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)


def main() -> None:
    exporter = ConsoleSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    SweagentInstrumentor().instrument(tracer_provider=provider)

    hooks = CombinedRunHooks()
    prob = MagicMock()
    prob.id = "demo-instance"
    prob.get_problem_statement.return_value = "Example task description"
    hooks.on_instance_start(
        index=0,
        env=MagicMock(),
        problem_statement=prob,
    )
    hooks.on_instance_completed(
        result=AgentRunResult(
            info=AgentInfo(exit_status="done"),
            trajectory=[],
        )
    )
    SweagentInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
