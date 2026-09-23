# LoongSuite A2A Instrumentation

OpenTelemetry instrumentation for the official
[A2A (Agent2Agent) Python SDK](https://github.com/a2aproject/a2a-python)
(`a2a-sdk`).

This package adds an ARMS gen-ai **AGENT** span around each server-side agent
turn, bracketing the user's `AgentExecutor.execute` implementation so that all
downstream work (the SDK's own transport/request-handler spans, plus any LLM
or tool instrumentation) nests underneath a single `invoke_agent` span with a
shared trace id.

It is **complementary** to the tracing already built into `a2a-sdk`
(`a2a.utils.telemetry`): the SDK traces protocol plumbing under the
`a2a-python-sdk` instrumenting module with generic span names and no gen-ai
semantic conventions, and it does not wrap the user's `execute` method. This
package supplies exactly that missing gen-ai agent boundary.

## Installation

```bash
pip install loongsuite-instrumentation-a2a
```

## Usage

```python
from opentelemetry.instrumentation.a2a import A2AInstrumentor

A2AInstrumentor().instrument()
```

Instrumentation covers both `AgentExecutor` subclasses that already exist when
`instrument()` is called and any defined afterwards (via an
`__init_subclass__` hook installed on `AgentExecutor`).

## Scope: agent execution, not the A2A protocol

This package instruments the **agent execution boundary** only (the
`invoke_agent` AGENT span). It does not model the A2A wire protocol
(client/server method spans such as `SendMessage` / `GetTask`); that belongs in
a dedicated protocol instrumentation and can follow separately, tracking the
[A2A semantic conventions draft](https://github.com/open-telemetry/semantic-conventions-genai/pull/195).
Where that draft already names stable protocol context available at the
execution boundary — the task id and task state — this package attaches it to
the AGENT span using the draft's `a2a.task.id` / `a2a.task.state` keys so the
execution span can be correlated with protocol telemetry.

## Content capture

The user's input message is recorded on `gen_ai.input.messages` **only when**
content capture is explicitly enabled through the shared GenAI switch used by
every loongsuite instrumentation. An absent or invalid value defaults to
`NO_CONTENT` (no message content), so sensitive prompts are never exported
without opt-in:

```bash
# Record message content on spans (default is NO_CONTENT):
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```
