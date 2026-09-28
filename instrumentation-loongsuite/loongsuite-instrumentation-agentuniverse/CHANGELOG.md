# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial release of `loongsuite-instrumentation-agentuniverse`: an independent
  OpenTelemetry instrumentation for the
  [agentUniverse](https://github.com/alipay/agentUniverse) multi-agent framework
  (`agentUniverse >= 0.0.19`), aligned with the capabilities of the framework's
  own instrumentation but implemented from scratch, without importing,
  delegating to or patching any of its telemetry classes.
- All three layers of an agent call tree are instrumented, one span per layer:
  - AGENT: `au.agent.{source}`, `au.span.kind=agent`, `au.agent.name/input/
    output/status/duration/pair_id/streaming/first_token.duration`,
    `au.agent.error.*`, `au.trace.caller_*`, `au.agent.usage.*`, and the nine
    `agent_*` metrics;
  - LLM: `au.llm.{source}`, `au.span.kind=llm`, `au.llm.name/channel_name/
    input/output/llm_params/streaming/duration/status/first_token.duration`,
    `au.llm.error.*`, `au.trace.caller_*`, `au.llm.usage.*`, and the nine
    `llm_*` metrics;
  - TOOL: `au.tool.{source}`, `au.span.kind=tool`, `au.tool.name/input/output/
    duration/status/pair_id`, `au.tool.error.*`, `au.trace.caller_*`,
    `au.tool.usage.*`, and the eight `tool_*` metrics.
- The ARMS/LoongSuite GenAI semantic conventions are added to those same spans
  through the shared `ExtendedTelemetryHandler` of `opentelemetry-util-genai`:
  `gen_ai.operation.name` (`invoke_agent` / `chat` / `execute_tool`),
  `gen_ai.span.kind`, `gen_ai.framework=agentuniverse`, the layer identity
  (`gen_ai.agent.name`, `gen_ai.request.model` / `gen_ai.provider.name`,
  `gen_ai.tool.name` / `gen_ai.tool.type` / `gen_ai.tool.call.id`),
  `gen_ai.usage.*`, `gen_ai.response.time_to_first_token`, the LLM's
  `gen_ai.response.finish_reasons`, the tool call arguments and result, and the
  conventional message content with capture on. Span names stay the framework's
  `au.*` form, through a name-preserving span proxy.
- Framework behaviour beyond spans is reproduced: sync and async calls on all
  three layers, streaming first-token timing for the agent (`output_stream`
  queues) and the LLM, deferred stream finalization, `LLMOutput.usage`
  aggregation onto the parent agent span, `ConversationMemoryModule` recording
  for agent and tool calls, the invocation chain that produces
  `au.trace.caller_*`, the `_llm_plugins` extension point, and error status,
  error attributes and error metrics.
- Content capture is governed by the shared GenAI util: the standard
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` switch (default
  `NO_CONTENT`) gates every content carrier, evaluated per call. With capture
  off, no `au.*.input/output` and no `gen_ai.*` content attribute is written on
  any layer, `au.llm.llm_params` keeps only scalar request parameters, and
  `au.*.error.message` (and the span status description) carry the exception
  type alone. With capture on, both the `au.*` and the `gen_ai.*` content
  carriers are written.
- Telemetry is fail-safe: every attribute write, metric record, memory call,
  invocation-chain push and serialization is guarded, so a telemetry failure is
  logged and swallowed while the instrumented call runs unchanged, and a user
  exception -- `asyncio.CancelledError` included -- is re-raised after the span
  has been closed with an error status. A layer that cannot be constructed calls
  the original function directly.
- Tests run against a real `agentUniverse` 0.0.19.1 install on Python 3.12 with
  no stand-in: 77 pass, one is skipped. They include a baseline matrix that runs
  the same workload under the framework's own three instrumentors and under this
  package and compares span tree, span names, kinds, status, `au.*` key set,
  all 26 metrics with their labels and values, non-zero token usage, positive
  first-token durations, error paths and conversation-memory side effects. The
  metric families are compared as exact sets -- nine agent, nine LLM and eight
  tool families, none missing and none extra -- and the tool content carriers
  are asserted directly in both capture modes.
- Mutation checks document that the tests are load-bearing. Neutering the agent,
  LLM or tool wrapper (29 / 27 / 16 failing tests), the transactional takeover
  (32), rollback/restore (3), the privacy switch (9), stream finalization (7),
  token aggregation (6) or the session probe (1) each turns the corresponding
  tests red, and the suite is green again once restored.
- The package advertises its metrics in the packaging metadata
  (`supports_metrics` in `pyproject.toml`, `_supports_metrics` in
  `package.py`), so the generated `instrumentation-loongsuite/README.md`
  lists it with `Metrics support = Yes`.

### Changed

- **The instrumentor performs its own Agent, LLM and Tool telemetry; it does not
  import, delegate to or patch the framework's instrumentation.** It takes over
  the six module-level wrapper globals of `agentuniverse.base.annotation.trace`
  -- `_agent_wrapper_sync` / `_agent_wrapper_async`, `_llm_wrapper_sync` /
  `_llm_wrapper_async`, `_tool_wrapper_sync` / `_tool_wrapper_async`, the same
  extension points the `@trace_agent`, `@trace_llm` and `@trace_tool` decorators
  read at call time -- and creates its own spans, attributes and metrics. The
  framework's `AgentInstrumentor`, `LLMInstrumentor`, `ToolInstrumentor`, their
  `*SpanAttributesSetter` classes, span managers and metric recorders are never
  imported or called at runtime.
  - An earlier revision of this package was a compatibility bridge that delegated
    span creation to those instrumentors and patched their attribute setters.
    That architecture has been replaced by this independent implementation, which
    removes the runtime dependency on the framework's telemetry classes entirely.
  - Reusable *business* helpers of the framework are still used: the
    `_get_agent_info` / `_get_llm_info` / `_get_tool_info` source-name helpers,
    `ConversationMemoryModule`, `Monitor` for the invocation chain,
    agentUniverse's token-usage bookkeeping, and the Agent / LLM / Tool data
    types.
- Installing is transactional and reversible. The six extension points are
  snapshotted before they are replaced; a failure while replacing any of them
  rolls the already-replaced ones back, so a partially installed state cannot
  double-wrap a layer. `uninstrument()` restores each saved object, but only if
  the extension point still holds this instrumentor's wrapper -- a wrapper that
  something else replaced in the meantime is left alone.
- **Coexistence with the framework's own instrumentation is a takeover, not a
  delegation.** If the application already started `AgentInstrumentor`,
  `LLMInstrumentor` and/or `ToolInstrumentor` (for example through
  `TelemetryManager.init_from_config()`), LoongSuite replaces their wrapper
  globals with its own for as long as it is installed, so a call is never
  wrapped twice and never produces two spans or two metric data points for one
  call. The saved bound methods are restored on `uninstrument()`, after which the
  framework's instrumentation works again. The framework's instrumentor objects
  and their `BaseInstrumentor` state are never called or modified, only the
  module-level extension points, reversibly. Every assembly order, and a mixed
  setup with only some layers pre-activated, is covered by tests.
- Two differences from the framework's own instrumentation are intentional and
  pinned by tests:
  - **Content capture.** The framework's setters write `au.*` content
    unconditionally; this package treats the shared capture switch as a privacy
    guarantee on every span it produces.
  - **Streamed LLM usage.** The framework's streaming path adds the streamed
    usage to the LLM span twice, so the parent agent span and the
    `agent_*_tokens` metrics receive double the real usage (20 tokens for a
    stream whose usage is 10). This package aggregates streamed usage exactly
    once.
- Session propagation stays where the framework defines it:
  `au.trace.session.id` and `AUSessionPropagator` come from agentUniverse's
  `TelemetryManager.init_from_config()`. This instrumentation activates only the
  Agent, LLM and Tool instrumentation and never touches global propagator or
  tracer provider state, so it does not register the session processor or the
  propagator. When an application does register them through `TelemetryManager`,
  the spans this package produces carry `au.trace.session.id` and the propagator
  injects and extracts `AU-SessionId` / `auSessionId`; that path is verified in
  an isolated process, while the main process asserts that instrumenting leaves
  the global tracer provider and propagator untouched.
- `wrapt` is not a declared dependency. The instrumentation patches the
  module-level extension points with `setattr`; the only remaining consumer,
  `opentelemetry-instrumentation`, already depends on it.
- `requires-python` is bounded to `>=3.10,<3.13` and the Python 3.13 classifier
  is dropped, matching the tox matrix: agentUniverse's pins (`numpy<2`,
  `grpcio==1.63.0`, `pyarrow<17`) have no cp313 wheels, so the `instruments`
  extra cannot be installed there.
