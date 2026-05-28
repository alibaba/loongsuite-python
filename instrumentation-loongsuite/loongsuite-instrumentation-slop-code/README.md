# LoongSuite slop-code-bench Instrumentation

OpenTelemetry instrumentation for the [slop-code-bench](https://github.com/SprocketLab/slop-code-bench) benchmark orchestrator.

## Span Tree

```
ENTRY  "enter_ai_application_system"
└── CHAIN  "chain {problem_name}"
    ├── ENTRY  "enter_ai_application_system"    [checkpoint worker]
    │   └── TASK  "run_task {checkpoint_name}"
    │       └── AGENT  "invoke_agent {agent_name}"
    │           ├── STEP  "react step"          [MiniSWE only]
    │           │   └── TOOL  "execute_tool bash"
    │           └── ...
    ├── ENTRY  "enter_ai_application_system"    [checkpoint worker]
    │   └── TASK  "run_task {checkpoint_name}"
    │       └── AGENT  "invoke_agent {agent_name}"
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
