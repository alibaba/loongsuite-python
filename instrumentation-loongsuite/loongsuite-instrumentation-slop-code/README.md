# LoongSuite slop-code-bench Instrumentation

OpenTelemetry instrumentation for the [slop-code-bench](https://github.com/SprocketLab/slop-code-bench) benchmark orchestrator.

## Span Tree

```
ENTRY  "slop-code.enter"
└── CHAIN  "workflow.{problem_name}"
    ├── TASK  "task.{checkpoint_name}"
    │   └── AGENT  "agent.{agent_type}"
    │       ├── STEP  "react.step.{N}"          [MiniSWE only]
    │       └── ...
    ├── TASK  "task.{checkpoint_name}"
    │   └── AGENT  "agent.{agent_type}"
    └── ...
LLM  "chat {model_name}"                       [Rubric Judge]
```

## Installation

```bash
pip install loongsuite-instrumentation-slop-code
```

## Usage

```python
from opentelemetry.instrumentation.slop_code import SlopCodeInstrumentor

SlopCodeInstrumentor().instrument()
```
