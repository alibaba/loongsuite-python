# LoongSuite WebArena Instrumentation

OpenTelemetry instrumentation for WebArena benchmark runs.

## Installation

```bash
pip install loongsuite-instrumentation-webarena
```

## Usage

```python
from opentelemetry.instrumentation.webarena import WebArenaInstrumentor

WebArenaInstrumentor().instrument()
```

For GenAI semantic conventions and span-only message content capture, set:

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```
