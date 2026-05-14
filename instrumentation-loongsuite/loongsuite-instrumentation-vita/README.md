# LoongSuite VitaBench Instrumentation

OpenTelemetry instrumentation for the VitaBench multi-domain simulation framework.

## Installation

```bash
pip install loongsuite-instrumentation-vita
```

## Usage

```python
from opentelemetry.instrumentation.vita import VitaInstrumentor

VitaInstrumentor().instrument()
```

For GenAI semantic conventions and span-only message content capture, set:

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```

## VitaBench With DashScope

VitaBench posts directly to the `base_url` configured in `models.yaml`, so the
DashScope OpenAI-compatible endpoint must include `/chat/completions`. The API
key must be supplied in the `Authorization` header.

```yaml
default:
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
  temperature: 0.0
  max_input_tokens: 8192
  headers:
    Content-Type: "application/json"
    Authorization: "Bearer ${OPENAI_API_KEY}"
models:
  - name: qwen3.6-plus
    max_tokens: 1024
    max_input_tokens: 8192
```

See `examples/vitabench-dashscope` for a runnable setup used by the Kubernetes
benchmark deployment.
