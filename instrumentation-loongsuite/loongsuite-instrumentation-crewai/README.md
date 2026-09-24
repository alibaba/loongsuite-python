# LoongSuite CrewAI Instrumentation

This library provides automatic instrumentation for [CrewAI](https://www.crewai.com/), a framework for orchestrating role-playing, autonomous AI agents.

## Installation

```bash
# Step 1: install LoongSuite distro
pip install loongsuite-distro

# Step 2 (Option C): install this instrumentation from PyPI
pip install loongsuite-instrumentation-crewai
```

## Usage

```python
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor

# Instrument CrewAI
CrewAIInstrumentor().instrument()

# Your CrewAI code here
from crewai import Agent, Task, Crew

agent = Agent(
    role='Data Analyst',
    goal='Extract actionable insights',
    backstory='Expert in data analysis',
    verbose=True
)

task = Task(
    description='Analyze the latest AI trends',
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

result = crew.kickoff()
```

## Telemetry

This instrumentation uses `ExtendedTelemetryHandler` from
`opentelemetry-util-genai` and emits:

- `enter_ai_application_system` spans for `Crew.kickoff` and Flow kickoff
  entry points (`gen_ai.span.kind=ENTRY`, `gen_ai.operation.name=enter`)
- `invoke_agent` spans for CrewAI task and agent execution
  (`gen_ai.span.kind=AGENT`, `gen_ai.operation.name=invoke_agent`)
- `execute_tool` spans for CrewAI tool execution
  (`gen_ai.span.kind=TOOL`, `gen_ai.operation.name=execute_tool`)

CrewAI-specific framework details are kept in `gen_ai.crewai.*` attributes,
such as `gen_ai.crewai.operation`, while GenAI message content capture follows
the shared util-genai controls. Message content capture is disabled by default;
set both environment variables below before process start to capture
`gen_ai.input.messages`, `gen_ai.output.messages`, system instructions, and
content-like CrewAI fields such as task descriptions or agent backstories:

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```

CrewAI has its own cloud tracing path. Set `CREWAI_TRACING_ENABLED=false` when
you only want LoongSuite/OpenTelemetry instrumentation for a process.

## Smoke Examples

When running from a source checkout, the `examples/crewai_smoke.py` script
exercises real non-streaming, streaming, and concurrent CrewAI calls. It reads
credentials from `DASHSCOPE_API_KEY` or `OPENAI_API_KEY` and defaults to
DashScope's OpenAI-compatible endpoint. When only `DASHSCOPE_API_KEY` is set,
the example also sets `OPENAI_API_KEY`, `OPENAI_API_BASE`, and
`DASHSCOPE_API_BASE` in the current process so CrewAI and LiteLLM can use the
OpenAI-compatible DashScope endpoint consistently.
Set `CREWAI_SMOKE_MODEL=openai/qwen-turbo` when validating tool-call paths
through LiteLLM's OpenAI-compatible DashScope provider.

```bash
loongsuite-instrument python \
  instrumentation-loongsuite/loongsuite-instrumentation-crewai/examples/crewai_smoke.py \
  --mode sync

loongsuite-instrument python \
  instrumentation-loongsuite/loongsuite-instrumentation-crewai/examples/crewai_smoke.py \
  --mode stream

loongsuite-instrument python \
  instrumentation-loongsuite/loongsuite-instrumentation-crewai/examples/crewai_smoke.py \
  --mode concurrent --concurrency 3
```

For local otel-gui verification, direct OTLP traces to the local backend and
let the example configure an HTTP exporter:

```bash
export OTEL_SERVICE_NAME=loongsuite-crewai-smoke
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:5173/v1/traces
export CREWAI_SMOKE_MANUAL_INSTRUMENT=true
export CREWAI_SMOKE_CONFIGURE_OTLP=true

python instrumentation-loongsuite/loongsuite-instrumentation-crewai/examples/crewai_smoke.py \
  --mode concurrent --concurrency 3
```

## Supported Versions

- CrewAI >= 0.80.0

## References

- [CrewAI Documentation](https://docs.crewai.com/)
- [OpenTelemetry Python Documentation](https://opentelemetry-python.readthedocs.io/)
