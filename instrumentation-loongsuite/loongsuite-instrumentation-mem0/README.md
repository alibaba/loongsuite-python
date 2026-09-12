# LoongSuite Mem0 Instrumentation

[![PyPI version](https://badge.fury.io/py/opentelemetry-instrumentation-mem0.svg)](https://badge.fury.io/py/opentelemetry-instrumentation-mem0)

Mem0 Python Agent provides observability for applications that use [Mem0](https://github.com/mem0ai/mem0) as a long‑term memory backend.  
This document shows how to install the Mem0 instrumentation, how to run a simple example, and what telemetry data you can expect.  
For details on usage and installation of LoongSuite and Jaeger, please refer to  
[LoongSuite Documentation](https://github.com/alibaba/loongsuite-python/blob/main/README.md).

## Installing Mem0 Instrumentation

```bash
# Step 1: install LoongSuite distro
pip install loongsuite-distro

# Step 2 (Option C): install this instrumentation from PyPI
pip install loongsuite-instrumentation-mem0

# Optional dependency for threading context propagation
pip install opentelemetry-instrumentation-threading
```

## Collect Data

### Example Application

Create a simple `demo.py` that uses Mem0:

```python
from mem0 import Memory

memory = Memory()
memory.add("User likes Python programming", user_id="user123")
results = memory.search("What does the user like?", user_id="user123")
print(results)
```

### Setting Environment Variables

Configure OpenTelemetry exporters before running the example:

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=<trace_endpoint>
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_SERVICE_NAME=mem0-demo

# Enable GenAI experimental semantic conventions (required for GenAI content/event features)
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental

# (Optional) Capture message content – may contain sensitive data
# Must be one of: NO_CONTENT | SPAN_ONLY | EVENT_ONLY | SPAN_AND_EVENT
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY

# (Optional) Emit GenAI events (LogRecord). Requires a LoggerProvider exporter in your app.
export OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT=true
```

### Recommended: Using loongsuite-instrument

Start your application with `loongsuite-instrument` (recommended in root README):

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
export OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT=true

loongsuite-instrument \
    --traces_exporter console \
    python demo.py
```

### Results

In the backend (console, Jaeger, or LoongSuite), you should see:

- Spans representing Mem0 `Memory` / `MemoryClient` calls (e.g., `add`, `search`)
- Child spans for Vector Store, Graph Store, and Reranker operations (when internal phases are enabled)
- Attributes that describe the operation, user/session identifiers, providers, and result statistics

## Configuration

You can control the Mem0 instrumentation using environment variables.

### Core Settings

| Environment Variable                                      | Default | Description                                                                 |
|-----------------------------------------------------------|---------|-----------------------------------------------------------------------------|
| `OTEL_INSTRUMENTATION_MEM0_ENABLED`                       | `true`  | Enable or disable the Mem0 instrumentation entirely.                       |
| `OTEL_INSTRUMENTATION_MEM0_INNER_ENABLED`                 | `false` | Enable internal phases (Vector Store, Graph Store, Rerank).              |
| `OTEL_SEMCONV_STABILITY_OPT_IN`                           | *(empty)* | Set to `gen_ai_latest_experimental` to enable GenAI experimental semantics (required for content/event). |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`      | `NO_CONTENT` | Content capturing mode: `NO_CONTENT`, `SPAN_ONLY`, `EVENT_ONLY`, `SPAN_AND_EVENT` (may contain PII/sensitive data). |
| `OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT`                   | `false` | Emit GenAI events (`LogRecord`). Requires configuring a `LoggerProvider`. |

### Configuration Examples

```bash
# Enable internal phases (Vector/Graph/Reranker)
export OTEL_INSTRUMENTATION_MEM0_INNER_ENABLED=true

# Enable content capture (be careful with sensitive data)
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT

# Enable event emission (requires LoggerProvider exporter)
export OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT=true
```

## Semantic Conventions Status

Mem0 instrumentation ships a set of semantic attributes and span names in
`semconv.py` that are tailored for Memory / Vector / Graph / Rerank operations.
These conventions are **not yet part of the upstream OpenTelemetry semantic
conventions** and should be treated as experimental.

## Compatibility

- Python: `>= 3.8, <= 3.13`
- Mem0 / `mem0ai`: `>= 1.0.0`
- OpenTelemetry API: `>= 1.20.0`

## License

Apache License 2.0

## Issues & Support

If you encounter problems or have feature requests, please open an issue in the  
[loongsuite-python GitHub repository](https://github.com/alibaba/loongsuite-python/issues).

## Related Resources

- [Mem0 Documentation](https://docs.mem0.ai/)
- [OpenTelemetry Python](https://opentelemetry-python.readthedocs.io/)
- [Gen‑AI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)