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

"""Session propagation probe, run as its own process.

``TelemetryManager.init_from_config`` may only run once per process and it takes
over the process-wide tracer provider and text map, so the session test runs it
here and reports what it observed as JSON on stdout. The parent test asserts on
that JSON.

Everything in this probe is real: ``TelemetryManager`` and the default
``SessionSpanProcessor``/``AUSessionPropagator`` it installs, a real ``Agent``
subclass and a real ``@trace_llm`` method. Only the span exporter is this
probe's own, so the spans can be read back.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault(
    "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
)

import session_probe_support  # noqa: E402
from agentuniverse.agent.agent import Agent  # noqa: E402
from agentuniverse.agent.agent_model import AgentModel  # noqa: E402
from agentuniverse.base.annotation.trace import trace_llm  # noqa: E402
from agentuniverse.base.config.application_configer.app_configer import (  # noqa: E402
    AppConfiger,
)
from agentuniverse.base.config.application_configer.application_config_manager import (  # noqa: E402
    ApplicationConfigManager,
)
from agentuniverse.base.tracing.au_trace_manager import (
    AuTraceManager,  # noqa: E402
)
from agentuniverse.base.tracing.otel.telemetry_manager import (  # noqa: E402
    TelemetryManager,
)
from agentuniverse.llm.llm_output import LLMOutput, TokenUsage  # noqa: E402

from opentelemetry import propagate, trace  # noqa: E402

SESSION = "session-abc123"
CARRIER_SESSION = "session-from-carrier"


class ProbeLLM:
    """A real ``@trace_llm`` method, which is all a call needs."""

    name = "session_llm"
    channel_name = "test_channel"

    @trace_llm
    def call(self, prompt: str, **kwargs: Any) -> LLMOutput:
        return LLMOutput(
            text=f"llm:{prompt}",
            usage=TokenUsage(text_in=1, text_out=2),
            finish_reason="stop",
        )


class ProbeAgent(Agent):
    """A real agent whose body is one real LLM call."""

    def input_keys(self) -> list:
        return ["input"]

    def output_keys(self) -> list:
        return ["output"]

    def parse_input(self, input_object: Any, agent_input: dict) -> dict:
        return agent_input

    def parse_result(self, agent_result: dict) -> dict:
        return agent_result

    def execute(self, input_object: Any, agent_input: dict) -> dict:
        output = ProbeLLM().call(prompt=input_object.get_data("input"))
        return {"output": output.text}


def main() -> None:
    ApplicationConfigManager().app_configer = AppConfiger()

    TelemetryManager().init_from_config(
        {
            "service_name": "agentuniverse-session-probe",
            "processors": [
                {
                    "class": (
                        "opentelemetry.sdk.trace.export.SimpleSpanProcessor"
                    ),
                    "exporter": {
                        "class": "session_probe_support.RecordingExporter"
                    },
                }
            ],
            "propagators": [],
            "metric_readers": [],
            "instrumentations": [],
        }
    )

    from opentelemetry.instrumentation.agentuniverse import (
        AgentUniverseInstrumentor,
    )

    instrumentor = AgentUniverseInstrumentor()
    instrumentor.instrument()

    AuTraceManager().set_session_id(SESSION)
    agent = ProbeAgent()
    agent.agent_model = AgentModel(info={"name": "session_agent"})
    output = agent.run(input="hello")

    carrier: dict = {}
    propagate.get_global_textmap().inject(carrier)
    propagate.get_global_textmap().extract({"AU-SessionId": CARRIER_SESSION})

    report = {
        "session": SESSION,
        "carrier_session": CARRIER_SESSION,
        "output": output.get_data("output"),
        "tracer_provider": type(trace.get_tracer_provider()).__name__,
        "spans": [
            {
                "name": span.name,
                "attributes": {
                    key: value
                    for key, value in (span.attributes or {}).items()
                    if key.startswith(("au.", "gen_ai."))
                },
            }
            for span in session_probe_support.SPANS
        ],
        "carrier": carrier,
        "session_after_extract": AuTraceManager().get_session_id(),
    }
    instrumentor.uninstrument()
    print("PROBE_JSON=" + json.dumps(report))


if __name__ == "__main__":
    main()
