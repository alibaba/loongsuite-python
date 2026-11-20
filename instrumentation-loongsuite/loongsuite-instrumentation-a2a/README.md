# LoongSuite A2A Instrumentation

OpenTelemetry instrumentation for the official
[A2A (Agent2Agent) Python SDK](https://github.com/a2aproject/a2a-python)
(`a2a-sdk`).

This package adds an ARMS gen-ai **AGENT** span around each server-side agent
turn, bracketing the user's `AgentExecutor.execute` implementation so all
downstream work (the SDK's own transport/request-handler spans, plus any LLM
or tool instrumentation) nests underneath a single `invoke_agent` span with a
shared trace id.

The AGENT span is owned end-to-end by the shared
[`opentelemetry-util-genai`](../../../util/opentelemetry-util-genai)
`ExtendedTelemetryHandler`: this instrumentation builds an
`InvokeAgentInvocation` at the executor boundary and drives it through
`start_invoke_agent` / `stop_invoke_agent` / `fail_invoke_agent`. It never
starts, ends, records errors on, or sets attributes on the AGENT span
itself and adds no extra structural span, matching the Hermes agent
instrumentation; the handler's `invoke_agent` span is the single
executor-boundary span.

It is **complementary** to the tracing already built into `a2a-sdk`
(`a2a.utils.telemetry`): the SDK traces protocol plumbing under the
`a2a-python-sdk` instrumenting module with generic span names and no gen-ai
semantic conventions, and it does not wrap the user's `execute` method. This
package supplies exactly that missing gen-ai agent boundary.

## Installation

```bash
pip install loongsuite-instrumentation-a2a
```

`opentelemetry-util-genai` is a declared runtime dependency, so the
`ExtendedTelemetryHandler` support is present in a clean environment without an
editable test install.

## Usage

```python
from opentelemetry.instrumentation.a2a import A2AInstrumentor

A2AInstrumentor().instrument()
```

Instrumentation covers both `AgentExecutor` subclasses that already exist when
`instrument()` is called and any defined afterwards (via an
`__init_subclass__` hook installed on `AgentExecutor`).

## Scope: agent execution, not the A2A protocol

This package instruments the **server-side agent execution boundary** only (the
handler-owned `invoke_agent` AGENT span plus the structural `INTERNAL`
`a2a.execute` span). It does not model the A2A wire protocol (no client or
client/server method spans such as `SendMessage` / `GetTask`); that belongs in
a dedicated protocol instrumentation and can follow separately, tracking the
[A2A semantic conventions draft](https://github.com/open-telemetry/semantic-conventions-genai/pull/195).

Per that draft, the A2A `contextId` — which groups a multi-turn agent
conversation — is mapped to the standard `gen_ai.conversation.id` attribute
(via `InvokeAgentInvocation.conversation_id`). The remaining task context
available at the execution boundary is attached to the AGENT span using the
draft's `a2a.task.id` / `a2a.task.state` keys so the execution span can be
correlated with protocol telemetry.

## Content capture

The user's input message is handed to the shared GenAI util as
`input_messages`; whether the prompt text is exported is decided solely by the
shared content-capture switch used by every loongsuite instrumentation. An
absent or invalid value defaults to `NO_CONTENT` (no message content), so
sensitive prompts are never exported without opt-in:

```bash
# Record message content on spans (default is NO_CONTENT):
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
# Or on the details event only (no content on the span):
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=EVENT_ONLY
```

## Fail-safety

Every telemetry step (invocation construction, handler
start/stop/fail, attribute setting and error recording) is fail-safe. A
telemetry failure never blocks the executor, never alters the A2A result, and
never replaces the business exception: on failure the original exception is
re-raised unchanged.
