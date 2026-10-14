# LoongSuite agentUniverse Instrumentation

OpenTelemetry instrumentation for the
[agentUniverse](https://github.com/alipay/agentUniverse) multi-agent framework
(`agentUniverse`).

This is an **independent implementation aligned with the capabilities of
agentUniverse's own instrumentation**. It does not import, delegate to or patch
the framework's instrumentors, span-attribute setters, span managers or metric
recorders. Instead it takes over the three documented wrapper extension points
of `agentuniverse.base.annotation.trace` -- the same seam the framework's own
instrumentation uses -- and produces the Agent, LLM and Tool telemetry itself:

* `au.agent.{source}`, `au.llm.{source}` and `au.tool.{source}` spans
  (`SpanKind.INTERNAL`), with the same `au.*` attribute set, the same 26 metrics
  (9 agent, 9 LLM, 8 tool), streaming first-token timing, conversation-memory
  recording, invocation-chain propagation, token aggregation and error handling
  the framework's instrumentors emit, and
* the ARMS/LoongSuite GenAI semantic conventions (`gen_ai.*`) on those same
  spans, through the shared `ExtendedTelemetryHandler` every LoongSuite
  instrumentation uses.

One agent call tree therefore carries exactly one span per layer -- agent, LLM
request, tool call -- each with both the `au.*` and the `gen_ai.*` namespaces.

## Requirements

The instrumentation itself is pure Python and runs on Python 3.10+. It needs
the `agentUniverse` distribution to import, and agentUniverse 0.0.19 pins
`numpy<2`, `grpcio==1.63.0` and `pyarrow<17`, none of which publish cp313
wheels (numpy 1.x cannot build on 3.13 either). agentUniverse therefore
installs on Python 3.10-3.12 today, and the LoongSuite test matrix mirrors
that range.

`requires-python` is bounded accordingly (`>=3.10,<3.13`), so the declared
range, the classifiers and the tox environments all say the same thing. On
3.13 the `instruments` extra cannot be resolved, and this instrumentation is
inert without agentUniverse.

`agentUniverse >= 0.0.19` is declared as the `instruments` extra.

## Installation

```bash
pip install loongsuite-instrumentation-agentuniverse
```

## Usage

```python
from opentelemetry.instrumentation.agentuniverse import (
    AgentUniverseInstrumentor,
)

AgentUniverseInstrumentor().instrument()
```

One call instruments all three layers. Every agent, every `@trace_llm` method
and every tool is covered, including classes defined after `instrument()` is
called, because the decorators read the wrapper globals at call time.

`tracer_provider` and `meter_provider` may be passed to `instrument()`; both
default to the OTel globals. `uninstrument()` removes the instrumentation
again.

## What is instrumented

### Agent

| | |
| --- | --- |
| Span | `au.agent.{source}`, `SpanKind.INTERNAL` |
| Attributes | `au.span.kind=agent`, `au.agent.name`, `au.agent.input`, `au.agent.output`, `au.agent.status`, `au.agent.duration`, `au.agent.pair_id`, `au.agent.streaming`, `au.agent.first_token.duration`, `au.agent.error.type`, `au.agent.error.message`, `au.trace.caller_name`, `au.trace.caller_type`, `au.agent.usage.total_tokens/prompt_tokens/completion_tokens/detail_tokens` |
| Metrics | `agent_calls_total`, `agent_errors_total`, `agent_call_duration`, `agent_first_token_duration`, `agent_total_tokens`, `agent_prompt_tokens`, `agent_completion_tokens`, `agent_cached_tokens`, `agent_reasoning_tokens` |
| Behaviour | sync and async runs; streaming runs wrap the `output_stream` queue (`queue.Queue`, `queue.SimpleQueue` and `asyncio.Queue`) to time the first token; `ConversationMemoryModule` records the input and the result; the invocation chain carries the agent node; token usage of nested LLM calls is aggregated onto the agent span |

### LLM

| | |
| --- | --- |
| Span | `au.llm.{source}`, `SpanKind.INTERNAL` |
| Attributes | `au.span.kind=llm`, `au.llm.name`, `au.llm.channel_name`, `au.llm.input`, `au.llm.output`, `au.llm.llm_params`, `au.llm.streaming`, `au.llm.duration`, `au.llm.status`, `au.llm.first_token.duration`, `au.llm.error.type`, `au.llm.error.message`, `au.trace.caller_name`, `au.trace.caller_type`, `au.llm.usage.prompt_tokens/completion_tokens/total_tokens/detail_tokens` |
| Metrics | `llm_calls_total`, `llm_errors_total`, `llm_call_duration`, `llm_first_token_duration`, `llm_total_tokens`, `llm_prompt_tokens`, `llm_completion_tokens`, `llm_cached_tokens`, `llm_reasoning_tokens` |
| Behaviour | sync and async calls; a streaming call's span stays open until the returned iterator has been consumed, and is finalized exactly once (normal completion, error, or an early `close()` / `aclose()`); the real `LLMOutput.usage` is aggregated onto the parent agent; the `_llm_plugins` extension point of `agentuniverse.base.annotation.trace` is applied before the original call, so application plugins keep working |

### Tool

| | |
| --- | --- |
| Span | `au.tool.{source}`, `SpanKind.INTERNAL` |
| Attributes | `au.span.kind=tool`, `au.tool.name`, `au.tool.input`, `au.tool.output`, `au.tool.duration`, `au.tool.status`, `au.tool.pair_id`, `au.tool.error.type`, `au.tool.error.message`, `au.trace.caller_name`, `au.trace.caller_type`, `au.tool.usage.*` |
| Metrics | `tool_calls_total`, `tool_errors_total`, `tool_call_duration`, `tool_total_tokens`, `tool_prompt_tokens`, `tool_completion_tokens`, `tool_cached_tokens`, `tool_reasoning_tokens` |
| Behaviour | sync and async tools; `ConversationMemoryModule` records the input and the output; the invocation chain carries the tool node |

### LoongSuite GenAI conventions

The `gen_ai.*` namespace is written by the shared `ExtendedTelemetryHandler` on
the very same span. Because the handler normalises span names to its own
`gen_ai` form, the layer hands it a thin name-preserving proxy: the `au.*`
span name is kept, while the handler still writes its attributes, records its
metrics and ends the span.

| Layer | `gen_ai.*` identity and request attributes |
| --- | --- |
| Agent | `gen_ai.operation.name=invoke_agent`, `gen_ai.span.kind=AGENT`, `gen_ai.agent.name`, `gen_ai.framework=agentuniverse` |
| LLM | `gen_ai.operation.name=chat`, `gen_ai.span.kind=LLM`, `gen_ai.request.model`, `gen_ai.provider.name`, `gen_ai.request.temperature` (only when the caller set one), `gen_ai.framework=agentuniverse` |
| Tool | `gen_ai.operation.name=execute_tool`, `gen_ai.span.kind=TOOL`, `gen_ai.tool.name`, `gen_ai.tool.type=function`, `gen_ai.tool.call.id`, `gen_ai.framework=agentuniverse` |

On top of that the handler writes `gen_ai.usage.input_tokens/output_tokens/
total_tokens` from the same numbers the `au.*.usage.*` attributes carry,
`gen_ai.response.time_to_first_token` for agent and LLM calls, the LLM's
`gen_ai.response.finish_reasons`, the tool call carriers
`gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` and, with content
capture on, the conventional `gen_ai.input.messages` /
`gen_ai.output.messages` / `gen_ai.system_instructions`.

The handler additionally records its own client metrics on those spans,
`gen_ai.client.operation.duration` and `gen_ai.client.token.usage`, next to the
26 `au.*` metrics. The framework's own instrumentation has no equivalent of
those two.

## Coexistence with agentUniverse's own instrumentation

`agentUniverse` ships its own OpenTelemetry instrumentors (`AgentInstrumentor`,
`LLMInstrumentor`, `ToolInstrumentor`), started either directly or through
`TelemetryManager.init_from_config()`. Both implementations take over the same
six module-level wrapper globals, so they must not both be active.

This package is written for that: **LoongSuite takes priority while it is
installed, and the framework's instrumentation is restored afterwards.**

* `instrument()` first snapshots all six extension points
  (`_agent_wrapper_sync` / `_agent_wrapper_async`, `_llm_wrapper_sync` /
  `_llm_wrapper_async`, `_tool_wrapper_sync` / `_tool_wrapper_async`) as they
  are at that moment -- plain framework defaults, or bound methods of a native
  instrumentor the application already started.
* It then replaces all six with its own wrappers and never calls the previous
  wrapper, so no call can be wrapped twice and no second span or metric can be
  produced. This is what keeps a three-instrumentor application at exactly one
  span per layer when LoongSuite is added.
* `uninstrument()` puts the saved objects back, extension point by extension
  point. If something else replaced a wrapper while LoongSuite was installed,
  that new state is left untouched instead of being overwritten.
* The framework's instrumentor objects are never called and their
  `BaseInstrumentor` state is never modified: LoongSuite only overrides the
  module-level extension points, reversibly. A native instrumentor the
  application started stays "instrumented" as far as the framework is
  concerned and resumes working as soon as LoongSuite restores the wrappers.
* Installing is transactional. If any extension point cannot be replaced, the
  wrappers already installed in that attempt are rolled back to the snapshot
  and the application keeps using the original wrappers -- a partially
  installed state, which could double-wrap a layer, never survives an install.
* Instrument and uninstrument are sentinelled, so repeating either is a no-op,
  and mixing native and LoongSuite instrumentation in either order is covered
  by tests.

An application that prefers the framework's telemetry can keep using it alone;
this package adds no telemetry unless it is instrumented.

## Content capture and privacy

Content capture follows the shared GenAI switch, read per call from the same
configuration every LoongSuite instrumentation reads. An absent or invalid
value defaults to `NO_CONTENT`, so prompts and results are never exported
without opt-in:

```bash
# Record message content on spans (default is NO_CONTENT):
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```

| | capture off (default) | capture on (`SPAN_ONLY` / `SPAN_AND_EVENT`) |
| --- | --- | --- |
| `au.agent.input/output`, `au.llm.input/output`, `au.tool.input/output` | not written, on any layer | written |
| `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` | not written | written |
| structure (identity, caller, timing, status, pairing, usage, errors) and all 26 metrics | unchanged | unchanged |

Capture is all-or-nothing per layer: either the input and the output are both
on the span, or neither is. `EVENT_ONLY` counts as off for span content,
because this instrumentation emits no events.

Two attribute families need extra care, so they get the same treatment:

* **`au.llm.llm_params`** may carry prompt-like values. With capture off only
  the scalar request parameters that cannot hold content (for example
  `temperature`) are kept; with capture on the whole parameter mapping is
  written.
* **`au.*.error.message`** is derived from the failing exception. With capture
  off it is the exception type alone, and the span status description is the
  same string, because an exception message or a traceback can quote the values
  that flowed through the failing call. With capture on the message includes
  `str(error)`. The framework's own instrumentation writes the full traceback
  unconditionally.

### A difference that is intentional

The framework's own instrumentation writes `au.*` content carriers
unconditionally: its attribute setters have no notion of the capture switch, so
with content capture at its default `NO_CONTENT` a native-instrumented
application still exports the raw prompt and the raw result. This package
instead treats `NO_CONTENT` as a privacy guarantee for every span it produces.
That is the first of the two deliberate differences below.

## Session propagation

Session ID propagation (`au.trace.session.id`, `AUSessionPropagator`) is part of
agentUniverse's telemetry setup, not of the instrumentation: it comes from
`SessionSpanProcessor` and `AUSessionPropagator`, which
`TelemetryManager.init_from_config()` registers. This package activates only
the Agent, LLM and Tool instrumentation and never touches global propagator or
tracer provider state, so an application that needs session propagation should
use `TelemetryManager` (which can also load this instrumentation by class
path), or register the processor and the propagator itself:

```python
from agentuniverse.base.tracing.otel.telemetry_manager import TelemetryManager

TelemetryManager().init_from_config({
    "service_name": "my-agent-app",
    "instrumentations": [
        "opentelemetry.instrumentation.agentuniverse:AgentUniverseInstrumentor",
    ],
})
```

That path is verified for real: a test runs `TelemetryManager.init_from_config()`
with an in-memory exporter in a separate process, sets the session id through
the framework, and asserts that the agent and LLM spans both carry
`au.trace.session.id`, that injecting into a carrier writes `AU-SessionId` and
`auSessionId`, and that extracting such a carrier restores the session id.
Running it out of process keeps the one-shot global provider and propagator out
of the pytest process -- which the main process asserts too: instrumenting
through this package leaves `trace.get_tracer_provider()` and
`propagate.get_global_textmap()` untouched.

## Token usage

`agentuniverse.llm.llm_output.TokenUsage` is a pydantic-v1 model whose real
fields are `text_in`, `image_in`, `audio_in`, `cached_in`, `text_out`,
`image_out`, `audio_out`, `cached_out` and `reasoning_out`; `prompt_tokens`,
`completion_tokens`, `cached_tokens`, `reasoning_tokens` and `total_tokens`
are read-only derived properties. Constructing
`TokenUsage(prompt=3, completion=5, total=8)` therefore sets nothing at all
(pydantic drops the unknown fields, leaving every counter zero) --
`TokenUsage(text_in=3, text_out=5)` is the non-zero form.

Aggregation is verified end to end with a real `@trace_llm` method returning
that usage: the LLM span carries `au.llm.usage.total_tokens=8`,
`prompt_tokens=3`, `completion_tokens=5` plus the matching `gen_ai.usage.*`,
and the same numbers appear on the parent agent span
(`au.agent.usage.total_tokens=8`, ...) and in the `agent_tokens` metrics,
recorded once.

### A second intentional difference: streamed usage

On the framework's streaming LLM path, the streamed usage is added to the LLM
span's token entry twice -- once while the stream is consumed, once again when
the stream is finalized -- so the parent agent span and the `agent_*_tokens`
metrics receive double the real usage. That is an upstream double count.

This implementation aggregates streamed usage exactly once. The baseline test
pins both numbers against a real streamed call: the framework's parent sees
20 tokens for a stream whose real usage is 10, LoongSuite's sees 10. The LLM
span itself carries the real numbers on both paths.

## Divergence from the framework's own instrumentation

| Area | Framework's instrumentors | This package |
| --- | --- | --- |
| Span names, kinds, `au.*` attribute keys, metric names and labels | baseline | identical; a test compares a native run and a LoongSuite run of the same workload item by item |
| Content capture | always writes `au.*` content | honours the capture switch; `NO_CONTENT` writes no content on any layer (intentional) |
| Streamed LLM usage into the parent | double counted | counted once (intentional) |
| `au.*.error.message` | full traceback | `str(error)` when capture is on, the exception type when it is off |
| `au.llm.llm_params` | whole mapping | whole mapping when capture is on, safe scalar parameters when it is off |
| Session processor and propagator | registered by `TelemetryManager` | unchanged: not registered, not touched; works when the application registers them |

## Fail-safe telemetry

Instrumentation never changes agent behaviour. Every attribute write, metric
record, memory call, invocation-chain push and content serialization is
guarded: a failure inside telemetry is logged and swallowed, the instrumented
call still runs, and its own exception -- including `asyncio.CancelledError` --
is re-raised unchanged after the span has been closed with an error status. A
layer that cannot even be constructed calls the original function directly.

## Modules

| Module | Responsibility |
| --- | --- |
| `__init__.py` | `AgentUniverseInstrumentor`: snapshot, transactional takeover of the six extension points, rollback, restore, tracer/meter/handler assembly |
| `_common.py` | shared span lifecycle, serialization, privacy, GenAI identity, token bookkeeping, metrics and fail-safe helpers |
| `_agent.py` | the agent layer: span, attributes, nine metrics, streaming first token, memory, aggregation |
| `_llm.py` | the LLM layer: span, attributes, nine metrics, deferred stream finalization, plugin hook, usage |
| `_tool.py` | the tool layer: span, attributes, eight metrics, memory |

## Tests

The suite runs against a real `agentUniverse` 0.0.19.1 install (Python 3.12)
with no stand-in: 77 tests pass, one is skipped.

* **Baseline matrix.** The same workload is run twice -- once with the
  framework's three instrumentors, once with this package -- and the two runs
  are compared: span tree and parent relationships, span names, kinds and
  status, the `au.*` key set and value semantics, all 26 metrics with their
  labels and values -- the family sets are compared exactly, nine agent, nine
  LLM and eight tool families, no family missing and none extra -- non-zero
  token usage (`TokenUsage(text_in=3, text_out=5)`
  reaching the agent span as 8 / 3 / 5), positive first-token durations, error
  spans, and `ConversationMemoryModule` side effects. Dynamic values (durations,
  pair ids, UUIDs) are compared by type and sign, not by value.
* **Ownership and lifecycle.** Native-only, LoongSuite-only, all three native
  instrumentors plus LoongSuite, and mixed setups; one span per layer in every
  case; instrument/uninstrument order, double instrument, failed install
  rollback, wrapper identity restoration, and native wrappers resuming work
  after a restore.
* **Behaviour.** Sync and async paths on all three layers, streaming agent and
  LLM first-token timing (positive durations on the span and in the
  `*_first_token_duration` histogram), stream finalization on completion,
  error and early close, error status and error metrics, content capture in
  both modes with a secret-scanning test that looks at every attribute value of
  every span, and the `_llm_plugins` hook.
* **Session.** The `TelemetryManager` path described above, in an isolated
  process.

Mutation checks confirm the tests are load-bearing. Neuter one seam, run the
suite, and the tests that must notice go red: the agent, LLM or tool wrapper
(29, 27 and 16 failures), the transactional takeover (32), rollback/restore
(3), the privacy switch (9), stream finalization (7), token aggregation (6) and
the session probe (1). Restoring the file turns the suite green again, with no
failures left behind.

```bash
python -m pytest tests -v
```
