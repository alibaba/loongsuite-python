# LoongSuite Instrumentation for Strands Agents

OpenTelemetry instrumentation for [Strands Agents](https://github.com/strands-agents/sdk-python) (Python), part of the LoongSuite instrumentation suite.

This package leverages the Strands SDK's native hooks system to produce spans conforming to the OpenTelemetry GenAI semantic conventions as extended by LoongSuite.

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
- strands-agents >= 0.1.0
- opentelemetry-api ~= 1.37
