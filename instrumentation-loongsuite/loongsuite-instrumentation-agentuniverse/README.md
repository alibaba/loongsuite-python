# LoongSuite agentUniverse Instrumentation

OpenTelemetry instrumentation for the
[agentUniverse](https://github.com/agentuniverse-ai/agentUniverse) multi-agent framework
(`agentUniverse`).

This is an **independent implementation aligned with the capabilities of
agentUniverse's own instrumentation**. It does not import, delegate to or patch
the framework's instrumentors, span-attribute setters, span managers or metric
recorders. Instead it takes over the three documented wrapper extension points
of `agentuniverse.base.annotation.trace` -- the same seam the framework's own
instrumentation uses -- and produces the Agent, LLM and Tool telemetry itself:

* `au.agent.{source}`, `au.llm.{source}` and `au.tool.{source}` spans
  (`SpanKind.INTERNAL`), with the `au.*` compatibility attributes and 26 metrics
  (9 agent, 9 LLM, 8 tool), streaming first-token timing, conversation-memory
  recording, invocation-chain propagation, token aggregation and error handling
  the framework's instrumentors emit, and
* the ARMS/LoongSuite GenAI semantic conventions (`gen_ai.*`) on those same
  spans, through the shared `ExtendedTelemetryHandler` every LoongSuite
  instrumentation uses.

One agent call tree carries one span per agent, LLM request and tool call.
Stable text ReAct uses the framework's LangChain executor; its Agent, LLM and
Tool calls are recorded without requiring the alpha-only round hooks. The layer spans carry
both the `au.*` and the `gen_ai.*` namespaces.

## Requirements

The compatibility target is **agentUniverse 0.0.19.1**, Python 3.10-3.12.
Agent/LLM/Tool wrappers, native text ReAct, OpenAI JSON/SSE responses and public
reasoning use this stable release's actual APIs. Optional newer hooks do not
prevent stable activation. The alpha release is not required.

**Release blocker: the published 0.0.19.1 package cannot currently be installed
with this integration's OpenTelemetry dependencies.** It requires
`grpcio==1.63.0`, while the aligned OTel gRPC exporter requires
`grpcio>=1.63.2`. The framework requires both the gRPC and HTTP exporter
distributions, so choosing HTTP does not remove the dependency conflict.
An upstream dependency correction and a new compatible stable package are
required before normal installation can be accepted.

Runtime diagnostic tests use unchanged stable framework source with a visibly
versioned, metadata-only local dependency correction. This establishes API
behavior, not compatibility of the unmodified published wheel. Neither
`--no-deps` nor `skip_dep_check=True` is a supported installation workaround.
The framework's numpy/pyarrow requirements limit this matrix to Python 3.10-3.12.

## Installation

Normal installation is blocked as described above. The `instruments` extra
and runtime version check target `agentUniverse==0.0.19.1`; they do not silently
select a prerelease. Do not treat this unreleased candidate as ready to install
until the dependency blocker is resolved.

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
capture on, role-preserving `gen_ai.input.messages` /
`gen_ai.output.messages` / `gen_ai.system_instructions`.

The handler additionally records its own client metrics on those spans,
`gen_ai.client.operation.duration` and `gen_ai.client.token.usage`, next to the
26 `au.*` metrics. The framework's own instrumentation has no equivalent of
those two. In event-enabled modes, the Span-compatible name proxy preserves
the real trace/span IDs on the shared handler's LLM and Agent events.

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
while GenAI LLM events follow the shared handler and the application's log exporter configuration.

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
This deliberate difference is covered in both the native baseline and complex runtime tests.

## Session propagation

The explicit `session_id` on an Agent run is recorded as `gen_ai.session.id`
on that Agent and its nested LLM and Tool spans. This context remains isolated
through the stable native ReAct executor and concurrent async conversations,
without changing global providers or propagators.


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

The framework's `TokenUsage` sums disjoint text, image, audio, cache and
reasoning buckets. OpenAI-compatible provider usage instead includes cache
reads in input totals and reasoning in output totals. This instrumentation
observes the raw provider usage at conversion time and creates its own corrected
bookkeeping value; it returns the original framework objects unchanged.

For provider usage of 20 input and 10 output tokens, with 5 cached input tokens
and 3 reasoning output tokens, every LLM span records 20 / 10 / 30 rather than
25 / 13 / 38. The same corrected totals reach Agent spans, the native metric
families and the shared GenAI metrics. Cache reads remain a subset reported in
`gen_ai.usage.cache_read.input_tokens`; reasoning remains a separate public
message part and a native usage detail/metric. Usage-only stream tails replace
cumulative snapshots instead of being added again.

Direct user-created `TokenUsage` values without a provider payload retain their
framework bucket semantics. No provider values are guessed from response text.
The stable framework's native instrumentation counts stream usage twice onto
its parent. This integration aggregates each stream once.

## Divergence from the framework's own instrumentation

| Area | Framework's instrumentors | This package |
| --- | --- | --- |
| Span names, kinds, `au.*` attribute keys, metric names and labels | baseline | identical; a test compares a native run and a LoongSuite run of the same workload item by item |
| Content capture | always writes `au.*` content | honours the capture switch; `NO_CONTENT` writes no content on any layer (intentional) |
| Streamed LLM usage into the parent | counted twice in 0.0.19.1 | counted once |
| OpenAI cache/reasoning subsets | may be added to inclusive totals twice | corrected from the raw provider totals without mutating business results |
| `au.*.error.message` | full traceback | `str(error)` when capture is on, the exception type when it is off |
| `au.llm.llm_params` | source helper may replace configured temperature with -1 | actual call/channel temperature, including zero; safe scalar parameters when capture is off |
| Session processor and propagator | registered by `TelemetryManager` | globals unchanged; explicit Agent session ID is also mapped to GenAI attributes |

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
| `_step.py` | native ReAct round spans |
| `_hooks.py` | reversible provider conversion and native tool dispatcher hooks |
| `_usage.py` | provider totals and call-local usage capture |
| `_tool.py` | the tool layer: span, attributes, eight metrics, memory |

## Tests

The suite runs against the declared real framework version. Fast wrapper,
coexistence, privacy, error and lifecycle checks are supplemented by
`tests/runtime_app.py` and `tests/stable_runtime_provider.py`:

- One native sync and two concurrent streaming conversations, each with two
  user turns and four model decisions per turn.
- The stable native ReAct/AgentExecutor dispatches stock, price and
  multiplication tools, then produces the final answer. Arguments, actual
  results and second-turn history are verified in real SDK traffic and spans.
  Text ReAct does not supply provider tool-call IDs or parallel calls within
  one model round; telemetry does not invent either capability.
- Nonempty synthetic public reasoning with `reasoning_effort=high`, reasoning
  disabled, fragmented text/reasoning and usage-only SSE tails. Only the
  provider HTTP boundary is deterministic; SDK parsing, framework loops,
  tool dispatch and telemetry are real.
- Separate real stable-channel SDK calls verify raw tool-call JSON and
  fragmented tool arguments in SSE. This checks model-output mapping, without
  claiming that native text ReAct dispatch supplies provider tool-call IDs.
- `SPAN_ONLY` and `NO_CONTENT`, correct inclusive Token totals and metrics,
  original business result identity, isolated sessions and parent trees.
- Atomic installation failure/retry, later hook ownership, unread stream close,
  cross-task consumption, cancellation, collector faults and clean siblings.

Run from the repository root:

```bash
tox -c tox-loongsuite.ini -e py312-test-loongsuite-instrumentation-agentuniverse
```

The test matrix uses the exporters from the same pinned OpenTelemetry core
revision as the SDK. Published exporters constrain released SDK versions and
cannot be mixed with that development SDK. The normal tox installation remains
blocked on the official framework dependency conflict; diagnostic runs are
reported separately. Six alpha-API checks are inapplicable to the stable
release and do not count as executed stable feature coverage.
