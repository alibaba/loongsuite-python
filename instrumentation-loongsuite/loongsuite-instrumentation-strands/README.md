# LoongSuite Instrumentation for Strands Agents

OpenTelemetry instrumentation for [Strands Agents](https://github.com/strands-agents/sdk-python) (Python), part of the LoongSuite instrumentation suite.

This package leverages the Strands SDK's native hooks system to produce spans conforming to the OpenTelemetry GenAI semantic conventions as extended by LoongSuite.

The hook adapter delegates span lifecycle, attributes, content capture, and
metrics to `opentelemetry-util-genai`, keeping the framework-specific code
limited to translating Strands events.

## Installation

```bash
pip install loongsuite-instrumentation-strands
```

## Usage

### Auto-instrumentation (via `loongsuite-instrument`)

```bash
loongsuite-instrument python your_app.py
```

### Manual instrumentation

```python
from opentelemetry.instrumentation.strands import StrandsInstrumentor

StrandsInstrumentor().instrument()
```

## Span Hierarchy

```
invoke_agent <agent_name>       (INTERNAL, gen_ai.span.kind=AGENT)
  └── react step                (INTERNAL, gen_ai.span.kind=STEP)
        ├── chat <model>        (CLIENT, gen_ai.span.kind=LLM)
        └── execute_tool <name> (INTERNAL, gen_ai.span.kind=TOOL)
```

## Requirements

- Python >= 3.10
- strands-agents >= 1.50.2, < 2.0.0
- opentelemetry-api ~= 1.37

## Configuration

- `OTEL_INSTRUMENTATION_STRANDS_LLM_SPAN_MODE=auto|always|never` controls the
  framework LLM span. `auto` is the default and avoids it when the model object
  reports that a provider instrumentor is already active.
- `OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_NATIVE=true|false` controls suppression
  of duplicate Strands SDK-native spans. The default is `true` and does not
  suppress model-provider instrumentation.
- Standard LoongSuite content-capture settings control prompt and response
  attributes; content is not captured by default.
