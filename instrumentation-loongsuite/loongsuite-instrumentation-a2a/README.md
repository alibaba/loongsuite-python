# LoongSuite A2A instrumentation

This package instruments A2A **protocol calls**, following the experimental
[OpenTelemetry A2A proposal #195](https://github.com/open-telemetry/semantic-conventions-genai/pull/195)
at revision `842a8397e638a28e86ca6e98fbb4dab26bc0e0cd`. These conventions are still
under development and may change.

```python
from opentelemetry.instrumentation.a2a import A2AInstrumentor

# Configure an OpenTelemetry TracerProvider and exporter first.
A2AInstrumentor().instrument()
```

Install and enable the instrumentation in both services to get A2A protocol
attributes on both sides. Each service also needs its A2A SDK and its normal
framework instrumentation. Cross-process trace propagation is provided by the
transport instrumentations, not by this package.

## Trace shape and propagation

```text
AgentScope agent (framework instrumentation)
  tool: call remote agent
    SendMessage CLIENT (A2A)
      HTTP CLIENT (HTTPX instrumentation injects trace context)
        SendMessage SERVER (ASGI/FastAPI extracts context; A2A enriches its span)
          LangChain agent/chain (framework instrumentation)
            model / tool calls
```

For HTTP, install and enable `opentelemetry-instrumentation-httpx` in the calling
service and the appropriate server instrumentation, such as
`opentelemetry-instrumentation-asgi` or `opentelemetry-instrumentation-fastapi`,
in the receiving service. Those instrumentations handle the configured
propagator, including `OTEL_PROPAGATORS`, and own injection/extraction of
`traceparent`, `tracestate` and baggage. For example, with a configured tracer
provider:

```python
# Calling service
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

HTTPXClientInstrumentor().instrument()

# Receiving service: wrap the actual ASGI app once, or use its framework probe.
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

app = OpenTelemetryMiddleware(app)
```

A2A uses the current context established by the application or transport probe.
It enriches and renames a detectable HTTP SERVER span without creating or ending
a second SERVER span. Otherwise, it creates its own protocol SERVER span under
the current context. It never wraps HTTPX or HTTP endpoints, reads trace headers
to establish context, or writes propagation headers. With only A2A instrumentation
enabled, protocol spans are recorded in each process but remote trace continuity
is not provided. HTTP instrumentation can already connect services without this
package; A2A instrumentation adds protocol semantics.
`contextId` and `taskId` are business identifiers, not replacements for
`traceparent`.

Supported hooks include JSON-RPC clients in SDK 0.2.x/0.3.x and 1.x, HTTP+JSON
clients in SDK 1.x, and existing or subsequently defined `RequestHandler`
subclasses. Methods are recorded when implemented by the installed SDK:
`SendMessage`, `SendStreamingMessage`, `GetTask`, `CancelTask`, `SubscribeToTask`,
`ListTasks`, extended Agent Card and push-notification configuration operations.
Direct `AgentExecutor.execute()` calls are not protocol calls and emit no A2A span.

For custom transports and gRPC, enable the corresponding client and server
transport instrumentations for propagation. This package does not add gRPC
client protocol hooks. A detectable current gRPC SERVER span is enriched without
renaming it. Requests rejected before reaching a request handler (for example
invalid JSON or unsupported protocol versions) remain the responsibility of HTTP
instrumentation.

## Attributes

| Source | Attribute |
| --- | --- |
| Logical method | `a2a.method.name` (also the span name) |
| Request message id | `a2a.message.id` |
| Request referenced tasks | `a2a.message.reference_task_ids` |
| Request or response task id | `a2a.task.id` |
| Response task status | `a2a.task.state`, normalized to `TASK_STATE_*` |
| Request or response context id | `gen_ai.conversation.id` |
| Available Agent Card metadata | `gen_ai.agent.name`, `.description`, `.version` |
| Selected interface / incoming version header | `a2a.protocol.version` |
| Request tenant | `a2a.tenant` |
| Client endpoint | `server.address`, `server.port` |
| Failed call | `error.type` and ERROR status |

Fields are omitted when unavailable. The request message id is not overwritten
by the response message id. Protocol spans never capture message parts, prompts,
model outputs, exception messages, authorization headers, or raw URLs with query
strings. This applies in all GenAI content modes, including `SPAN_ONLY`.
Framework Agent/LLM/Tool content remains governed by GenAI util's capture policy.
No A2A metric instruments or `gen_ai.operation.name` are introduced.

## Streaming and failure isolation

Client spans cover response consumption; server spans cover response production.
Terminal message/status events end owned spans before the result reaches the
consumer. Exhaustion, explicit `aclose()`, cancellation, errors and finalization
also end spans at most once. Context is attached only during a pull or close and
restored before returning to user code, including consumption in another task.
Close streams explicitly when abandoning them before a terminal event.

Probe-only operations use LoongSuite GenAI util's `hook_advice`. The business
call executes once, preserves its original result/chunks/exceptions, and does
not depend on successful telemetry. A2A does not add protocol APIs to
`ExtendedTelemetryHandler`; higher-level framework spans continue to use it.

## Migration from executor spans

The initial implementation created `invoke_agent <ExecutorClass>` for each
executor. This is replaced by protocol CLIENT/SERVER spans to avoid double
counting when an executor runs an already instrumented agent framework. Enable
the corresponding framework instrumentation for Agent/LLM/Tool details. A custom
agent may use GenAI util explicitly at its real agent invocation boundary.

SDK internal function/queue spans are suppressed while this instrumentor is
enabled, even if the SDK was imported earlier. This only replaces the SDK's
module-local tracer and is restored by `uninstrument()`; the current transport
span, global tracing and the environment are unchanged. To retain native SDK spans for debugging, explicitly
set `OTEL_INSTRUMENTATION_A2A_SDK_ENABLED=true` **before importing the SDK**.
That debug mode intentionally retains SDK noise and is not a clean protocol tree.

## Tests

```bash
pytest instrumentation-loongsuite/loongsuite-instrumentation-a2a/tests
```

The suite uses real SDK clients, HTTP dispatchers, executors and streaming
serialization, plus fault injection. It uses a local ASGI transport and requires
no model credentials or network access. The ASGI test boundary isolates caller
context to model separate processes; only enabling both transport probes joins
the traces. HTTPX/ASGI test instrumentation is
included in `test-requirements.txt`. SDK 0.2.1 requires Python 3.13; SDK 0.3.0 can
exercise the package's Python 3.10 lane. Current SDK 1.x has its own Python version
requirements.
