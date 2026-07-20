# LoongSuite DeerFlow Instrumentation

LoongSuite instrumentation for [DeerFlow](https://github.com/bytedance/deer-flow) 2.x.

## Installation

Install DeerFlow from its official Git repository, then install this package:

```bash
pip install 'deerflow-harness @ git+https://github.com/bytedance/deer-flow.git@v2.0.0#subdirectory=backend/packages/harness'
pip install loongsuite-instrumentation-deerflow
```

The similarly named `deerflow-harness` package currently published on PyPI is
not the official DeerFlow 2.x distribution. This instrumentation deliberately
does not expose an `instruments` extra that would install it.

## Usage

```python
from opentelemetry.instrumentation.deerflow import DeerFlowInstrumentor

DeerFlowInstrumentor().instrument()
```

The instrumentor composes the LoongSuite LangChain and LangGraph
instrumentations, identifies DeerFlow-created graphs, and creates an `ENTRY`
span around Gateway runs and embedded `DeerFlowClient.stream()` calls. Direct
SDK graph invocation starts at the graph's `AGENT` span and does not create a
synthetic application entry.

The core execution tree is:

```text
ENTRY deerflow
└── AGENT lead-agent
    └── STEP
        ├── LLM
        └── TOOL task
            └── AGENT subagent:<name>
```

An existing host `ENTRY` takes precedence over the Gateway and embedded entry
points. The embedded iterator restores its call-time context around every
`next()` and `close()`, including cross-thread consumption, without leaking
that context back to the consumer.

`gen_ai.session.id` and `gen_ai.user.id` prefer existing baggage values over
DeerFlow identities. DeerFlow's request trace id is recorded as
`deerflow.trace.id`; it is a correlation field and never replaces the W3C
OpenTelemetry trace id. Gateway runs expose their real `deerflow.run.id`.
Embedded runs omit that field because DeerFlow's public stream API does not
expose its internally generated run id.

Message content is disabled by default and follows the existing experimental
GenAI content-capture switch.

`task` remains a normal LangChain tool. Operations performed in a sandbox are
visible when DeerFlow exposes them as LangChain tools, and work selected by a
skill remains visible through its downstream LLM and tool calls. This package
does not create separate lifecycle spans for sandbox allocation or reuse, skill
activation, or background memory work, and it does not patch DeerFlow's context
propagation internals.

## Compatibility

- DeerFlow `>=2,<3`, installed from the official source repository
- Python 3.12+
