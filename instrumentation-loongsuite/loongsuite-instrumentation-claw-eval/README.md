# LoongSuite Claw Eval Instrumentation

OpenTelemetry instrumentation for Claw Eval benchmark runs.

## Installation

```bash
pip install loongsuite-instrumentation-claw-eval
```

## Usage

```python
from opentelemetry.instrumentation.claw_eval import ClawEvalInstrumentor

ClawEvalInstrumentor().instrument()
```

For GenAI semantic conventions and span-only message content capture, set:

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```
