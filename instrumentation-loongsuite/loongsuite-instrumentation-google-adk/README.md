# LoongSuite Google ADK Instrumentation

Google ADK (Agent Development Kit) Python Agent provides comprehensive observability for Google ADK applications using OpenTelemetry.

## Features

- ✅ **Automatic Instrumentation**: Zero-code integration via `loongsuite-instrument`
- ✅ **Manual Instrumentation**: Programmatic control via `GoogleAdkInstrumentor`
- ✅ **GenAI Semantic Conventions**: Full compliance with OpenTelemetry GenAI standards
- ✅ **Comprehensive Spans**: `invoke_agent`, `chat`, `execute_tool`
- ✅ **Standard Metrics**: Operation duration and token usage
- ✅ **Content Capture**: Optional message and response content capture
- ✅ **Google ADK native instrumentation Compatible**: Works seamlessly with ADK native instrumentation

## Quick Start

```bash
# Step 1: install LoongSuite distro
pip install loongsuite-distro

# Step 2 (Option C): install instrumentation from PyPI
pip install loongsuite-instrumentation-google-adk

# App dependencies
pip install google-adk litellm

# Configure
export DASHSCOPE_API_KEY=your-api-key

# Run with auto instrumentation
loongsuite-instrument \
  --traces_exporter console \
  --service_name my-adk-app \
  python your_app.py
```

For details on LoongSuite and Jaeger setup, refer to [LoongSuite Documentation](https://github.com/alibaba/loongsuite-python-agent/blob/main/README.md).

## Installing Google ADK Instrumentation

```shell
# Step 1: install LoongSuite distro
pip install loongsuite-distro

# Step 2 (Option C): install this instrumentation from PyPI
pip install loongsuite-instrumentation-google-adk

# Google ADK and LLM Dependencies
pip install google-adk>=0.1.0
pip install litellm

# Demo Application Dependencies (optional, only if running examples)
pip install fastapi uvicorn pydantic

```

## Collect Data

Here's a simple demonstration of Google ADK instrumentation. The demo uses:

- A [Google ADK application](examples/main.py) that demonstrates agent interactions with multiple tools

### Running the Demo

> **Note**: The demo uses DashScope (Alibaba Cloud LLM service) by default. You need to set the `DASHSCOPE_API_KEY` environment variable.

#### Option 1: Using LoongSuite auto instrumentation

```bash
# Set your DashScope API key
export DASHSCOPE_API_KEY=your-dashscope-api-key

# Enable content capture (optional, for debugging)
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY

# Run with loongsuite instrumentation
loongsuite-instrument \
  --traces_exporter console \
  --service_name demo-google-adk \
  python examples/main.py
```

#### Option 2: Export to Jaeger

```bash
# Set your DashScope API key
export DASHSCOPE_API_KEY=your-dashscope-api-key

# Configure OTLP exporter
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc

# Run the application
loongsuite-instrument \
  --service_name demo-google-adk \
  python examples/main.py
```

#### Option 3: Local otel-gui smoke scenarios

`examples/otelgui_smoke.py` produces real non-streaming, SSE streaming, and
concurrent Google ADK calls for local trace validation.

```bash
export DASHSCOPE_API_KEY=your-dashscope-api-key
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:5173
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:5173/v1/traces
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
export OTEL_SERVICE_NAME=loongsuite-google-adk-smoke
export GOOGLE_ADK_SMOKE_CONFIGURE_OTLP=1
export GOOGLE_ADK_SMOKE_DISABLE_NATIVE_AGENT_SPAN=1

python examples/otelgui_smoke.py --scenario all
```

`GOOGLE_ADK_SMOKE_DISABLE_NATIVE_AGENT_SPAN=1` uses a private ADK telemetry
monkey-patch during smoke validation to remove ADK's native wrapper span, making
the LoongSuite GenAI span tree easier to inspect in otel-gui. Keep it limited to
local smoke tests because private ADK internals may change.

When using the local `loongsuite-otelgui-plugin-verify` helper, select the
GenAI util agent trace explicitly because Google ADK also emits an `invocation`
trace:

```bash
python /path/to/run_loongsuite_plugin_smoke.py \
  --repo-root /path/to/loongsuite-python-agent \
  --base-url http://127.0.0.1:5173 \
  --service-name loongsuite-google-adk-non-stream \
  --root-span-contains invoke_agent \
  --capture-message-content SPAN_ONLY \
  --expect-span-kind AGENT \
  --expect-span-kind LLM \
  --expect-span-kind TOOL \
  --expect-content \
  --env GOOGLE_ADK_SMOKE_CONFIGURE_OTLP=1 \
  --env GOOGLE_ADK_SMOKE_DISABLE_NATIVE_AGENT_SPAN=1 \
  --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:5173/v1/traces \
  --run "python examples/otelgui_smoke.py --scenario non-stream"
```

### Expected Results

The instrumentation will generate traces showing the Google ADK operations:

#### Tool Execution Span Example

```json
{
    "name": "execute_tool get_current_time",
    "context": {
        "trace_id": "xxx",
        "span_id": "xxx",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": "xxx",
    "start_time": "2025-10-23T06:36:33.858459Z",
    "end_time": "2025-10-23T06:36:33.858779Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.span.kind": "TOOL",
        "gen_ai.tool.name": "get_current_time",
        "gen_ai.tool.description": "xxx",
        "gen_ai.tool.call.arguments": "{xxx}",
        "gen_ai.tool.call.result": "{xxx}"
    },
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.37.0",
            "service.name": "demo-google-adk"
        },
        "schema_url": ""
    }
}
```

#### LLM Chat Span Example

```json
{
    "name": "chat qwen-max",
    "kind": "SpanKind.CLIENT",
    "attributes": {
        "gen_ai.operation.name": "chat",
        "gen_ai.span.kind": "LLM",
        "gen_ai.request.model": "qwen-max",
        "gen_ai.response.model": "qwen-max",
        "gen_ai.usage.input_tokens": 150,
        "gen_ai.usage.output_tokens": 45
    }
}
```

#### Agent Invocation Span Example

```json
{
    "name": "invoke_agent ToolAgent",
    "kind": "SpanKind.CLIENT",
    "attributes": {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.span.kind": "AGENT",
        "gen_ai.agent.name": "ToolAgent",
        "gen_ai.input.messages": "[{\"role\": \"user\", \"parts\": [{\"type\": \"text\", \"content\": \"What time is it?\"}]}]",
        "gen_ai.output.messages": "[{\"role\": \"assistant\", \"parts\": [{\"type\": \"text\", \"content\": \"The current time is 2025-11-27 14:36:33\"}]}]"
    }
}
```

### Viewing in Jaeger

After [setting up Jaeger](https://www.jaegertracing.io/docs/latest/getting-started/), you can visualize the complete trace hierarchy in the Jaeger UI, showing the relationships between Runner, Agent, LLM, and Tool spans  

## Configuration

### Environment Variables

The following environment variables can be used to configure the Google ADK instrumentation:

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_SEMCONV_STABILITY_OPT_IN` | Enable latest experimental GenAI semantic conventions | - |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Capture message content in traces (`NO_CONTENT`, `SPAN_ONLY`, `SPAN_AND_EVENT`) | `NO_CONTENT` |
| `DASHSCOPE_API_KEY` | DashScope API key (required for demo) | - |

### Programmatic Configuration

You can also configure the instrumentation programmatically:

```python
from opentelemetry.instrumentation.google_adk import GoogleAdkInstrumentor

# Configure the instrumentor
instrumentor = GoogleAdkInstrumentor()

# Enable instrumentation with custom configuration
instrumentor.instrument(
    tracer_provider=your_tracer_provider,
    meter_provider=your_meter_provider
)
```

## Supported Features

### Traces

The Google ADK instrumentation automatically creates traces for:

- **Agent Runs**: Complete agent execution cycles
- **Tool Calls**: Individual tool invocations
- **Model Interactions**: LLM requests and responses
- **Session Management**: User session tracking
- **Error Handling**: Exception and error tracking

### Metrics

The instrumentation follows the [OpenTelemetry GenAI Semantic Conventions for Metrics](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-metrics.md) and provides the following **standard client metrics**:

#### 1. `gen_ai.client.operation.duration` (Histogram)

Records the duration of GenAI operations in seconds.

**Instrument Type**: Histogram  
**Unit**: `s` (seconds)  
**Status**: Development

**Required Attributes**:
- `gen_ai.operation.name`: Operation being performed (e.g., `chat`, `invoke_agent`, `execute_tool`)
- `gen_ai.provider.name`: Provider name (e.g., `google_adk`)

**Conditionally Required Attributes**:
- `error.type`: Error type (only if operation ended in error)
- `gen_ai.request.model`: Model name (if available)

**Recommended Attributes**:
- `gen_ai.response.model`: Response model name
- `server.address`: Server address
- `server.port`: Server port

**Example Values**:
- LLM operation: `gen_ai.operation.name="chat"`, `gen_ai.request.model="gemini-pro"`, `duration=1.5s`
- Agent operation: `gen_ai.operation.name="invoke_agent"`, `gen_ai.request.model="math_tutor"`, `duration=2.3s`
- Tool operation: `gen_ai.operation.name="execute_tool"`, `gen_ai.request.model="calculator"`, `duration=0.5s`

#### 2. `gen_ai.client.token.usage` (Histogram)

Records the number of tokens used in GenAI operations.

**Instrument Type**: Histogram  
**Unit**: `{token}`  
**Status**: Development

**Required Attributes**:
- `gen_ai.operation.name`: Operation being performed
- `gen_ai.provider.name`: Provider name
- `gen_ai.token.type`: Token type (`input` or `output`)

**Conditionally Required Attributes**:
- `gen_ai.request.model`: Model name (if available)

**Recommended Attributes**:
- `gen_ai.response.model`: Response model name
- `server.address`: Server address
- `server.port`: Server port

**Example Values**:
- Input tokens: `gen_ai.token.type="input"`, `gen_ai.request.model="gemini-pro"`, `count=100`
- Output tokens: `gen_ai.token.type="output"`, `gen_ai.request.model="gemini-pro"`, `count=50`

**Note**: These metrics use **Histogram** instrument type (not Counter) and follow the standard OpenTelemetry GenAI semantic conventions. All other metrics (like `genai.agent.runs.count`, etc.) are non-standard and have been removed to ensure compliance with the latest OTel specifications.

### Semantic Conventions

This instrumentation follows the OpenTelemetry GenAI semantic conventions:

- [GenAI Spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md)
- [GenAI Agent Spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-agent-spans.md)
- [GenAI Metrics](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-metrics.md)



## Troubleshooting

### Common Issues

1. **Module Import Error**: If you encounter `No module named 'google.adk.runners'`, ensure that `google-adk` is properly installed:
   ```bash
   pip install google-adk>=0.1.0
   ```

2. **DashScope API Error**: If you see authentication errors, verify your API key is correctly set:
   ```bash
   export DASHSCOPE_API_KEY=your-api-key
   # Verify it's set
   echo $DASHSCOPE_API_KEY
   ```

3. **Instrumentation Not Working**: 
   - Check that the instrumentation is enabled and the Google ADK application is using the `Runner` class
   - Verify you see the log message: `Plugin 'opentelemetry_adk_observability' registered`
   - For manual instrumentation, ensure you call `GoogleAdkInstrumentor().instrument()` before creating the Runner

4. **Missing Traces**: 
   - Verify that the OpenTelemetry exporters are properly configured
   - Check the `OTEL_TRACES_EXPORTER` environment variable is set (e.g., `console`, `otlp`)
   - For OTLP exporter, ensure the endpoint is reachable


## References

- [OpenTelemetry Project](https://opentelemetry.io/)
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [GenAI Semantic Conventions](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/)
