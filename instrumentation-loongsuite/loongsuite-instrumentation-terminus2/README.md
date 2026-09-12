# LoongSuite Terminus2 Instrumentation

OpenTelemetry instrumentation for Terminus2 benchmark runs.

## Installation

```bash
pip install loongsuite-instrumentation-terminus2
```

## Usage

```python
from opentelemetry.instrumentation.terminus2 import Terminus2Instrumentor

Terminus2Instrumentor().instrument()
```

For GenAI semantic conventions and span-only message content capture, set:

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```
