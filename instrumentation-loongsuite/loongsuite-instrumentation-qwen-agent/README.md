# OpenTelemetry Qwen-Agent Instrumentation

OpenTelemetry instrumentation for the [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) framework.

## Installation

```bash
pip install opentelemetry-distro opentelemetry-exporter-otlp
opentelemetry-bootstrap -a install

pip install "qwen-agent >= 0.0.20"

pip install ./util/opentelemetry-util-genai

# Install this instrumentation (from the LoongSuite repo)
pip install ./instrumentation-loongsuite/loongsuite-instrumentation-qwen-agent
```

Published package name:

```bash
pip install loongsuite-instrumentation-qwen-agent
```

## Usage

### Auto-instrumentation

With `loongsuite-instrumentation-qwen-agent` installed, the `opentelemetry_instrumentor` entry point `qwen_agent` is registered for use with the OpenTelemetry distro.

```bash
opentelemetry-instrument \
    --traces_exporter console \
    python your_qwen_agent_app.py
```

### Manual instrumentation

```python
from opentelemetry.instrumentation.qwen_agent import QwenAgentInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from qwen_agent.agents import Assistant

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

QwenAgentInstrumentor().instrument(tracer_provider=provider)

bot = Assistant(
    llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
    name="my-assistant",
)
for _ in bot.run([{"role": "user", "content": "Hello!"}]):
    pass

QwenAgentInstrumentor().uninstrument()
```

## Configuration

### Export to an OTLP backend

```bash
export OTEL_SERVICE_NAME=my-qwen-agent-app
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=<trace_endpoint>
# Optional: metrics / logs if you configure exporters globally
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=<metrics_endpoint>

opentelemetry-instrument python your_app.py
```

### GenAI semantic conventions and content capture

```bash
# Enable experimental GenAI semantic conventions (recommended for this instrumentation)
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental

# Message content capture (same env vars as other GenAI instrumentations in this repo)
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
# EVENT_ONLY | SPAN_AND_EVENT | NO_CONTENT
```

## Supported components

| Area | Instrumented API | Span / operation |
|------|------------------|------------------|
| Agent | `Agent.run` | `invoke_agent` |
| LLM | `BaseChatModel.chat` | `chat` |
| ReAct | `Agent._call_llm` | `react step` (agents with tools) |
| Tools | `Agent._call_tool` | `execute_tool` |

`Agent.run_nonstream()` is not wrapped separately; it calls `run()` internally, so you still get a single `invoke_agent` span per run.

**Model backends** (inferred from `model_type` / class name): DashScope, OpenAI-compatible APIs, Azure OpenAI, and other Qwen-Agent–supported backends.

## Visualization

Export telemetry to:

- [Alibaba Cloud ARMS / Managed Service for OpenTelemetry](https://www.aliyun.com/product/xtrace)
- [AgentScope Studio](https://github.com/agentscope-ai/agentscope-studio) or any OTLP-compatible collector
- Any OpenTelemetry-compatible backend (Jaeger, Zipkin, etc.)

## License

Apache License 2.0
