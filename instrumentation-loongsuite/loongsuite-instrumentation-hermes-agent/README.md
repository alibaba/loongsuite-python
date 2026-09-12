# LoongSuite Hermes Agent Instrumentation

LoongSuite instrumentation for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

## Features

- **Agent tracing**: `invoke_agent Hermes` spans for top-level agent runs
- **ReAct step tracing**: `react step` spans for each Hermes reasoning round
- **LLM tracing**: model request/response spans with token usage and TTFT
- **Tool tracing**: `execute_tool` spans for Hermes tool calls
- **Content capture controls**: honors `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`

## Installation

```bash
# Step 1: install LoongSuite distro
pip install loongsuite-distro

# Step 2 (Option C): install this instrumentation from PyPI
pip install loongsuite-instrumentation-hermes-agent

# Optional app dependency
pip install hermes-agent
```

## Usage

### Auto-instrumentation

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY

loongsuite-instrument python your_hermes_app.py
```

### Manual instrumentation

```python
from opentelemetry.instrumentation.hermes_agent import HermesAgentInstrumentor
from run_agent import AIAgent

HermesAgentInstrumentor().instrument()

agent = AIAgent(
    model="qwen-turbo",
    provider="dashscope",
    quiet_mode=True,
)
result = agent.run_conversation("Hello from Hermes")

HermesAgentInstrumentor().uninstrument()
```

## Configuration

### Export to OTLP Backend

```bash
export OTEL_SERVICE_NAME=my-hermes-app
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=<trace_endpoint>
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=<metrics_endpoint>

loongsuite-instrument python your_hermes_app.py
```

### Content Capture

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental

# Capture content in spans only
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY

# Disable content capture
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=NO_CONTENT
```

## Supported Signals

- **AGENT**: top-level Hermes agent invocation
- **ENTRY**: AI application entry spans when Hermes resolves an entry source
  from `AIAgent.platform`, `HERMES_SESSION_SOURCE`, or the default `cli`
  source used by platform-less `AIAgent` instances. Every top-level
  instrumented `AIAgent.run_conversation` now emits `ENTRY` -> `AGENT`.
- **STEP**: Hermes ReAct step lifecycle
- **LLM**: synchronous and streaming model calls
- **TOOL**: Hermes tool execution, including tool call id, arguments, and result

## License

Apache License 2.0
