# LoongSuite MiniSweAgent Instrumentation

OpenTelemetry instrumentation for MiniSweAgent runs.

## Installation

```bash
pip install loongsuite-instrumentation-minisweagent
```

## Usage

```python
from opentelemetry.instrumentation.minisweagent import MiniSweAgentInstrumentor

MiniSweAgentInstrumentor().instrument()
```

For GenAI semantic conventions and span-only message content capture, set:

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```
