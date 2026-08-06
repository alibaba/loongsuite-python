# Upstream synchronization

This package follows the canonical Google GenAI instrumentation maintained in
[`open-telemetry/opentelemetry-python-genai`](https://github.com/open-telemetry/opentelemetry-python-genai/tree/main/instrumentation/opentelemetry-instrumentation-google-genai).

## Baseline

- Package release: `opentelemetry-instrumentation-google-genai==1.0b1`
- Repository tag:
  `opentelemetry-instrumentation-google-genai==1.0b1`
- Repository commit: `d168d3e6bd280ae131fd56f60a15b449626ab336`
- Supported SDK range: `google-genai>=1.32,<3`
- Upstream `main` snapshot audited after the tag:
  `e83bf7df909abb5c2e069ec6480b942638389706`
- OpenInference Google GenAI comparison snapshot:
  `a1392c50d2d5b20fb805c195fb6006c80d5a6106`

The provider hook modules keep the upstream file layout:

- `allowlist_util.py`
- `client_info.py`
- `custom_semconv.py`
- `dict_util.py`
- `embeddings.py`
- `generate_content.py`
- `instrumentor.py`
- `interactions.py`
- `message.py`
- `tool_call_wrapper.py`

Upstream tests for generation, streaming, embeddings, interactions,
configuration allowlists, finish reasons, and automatic function calls are
reused in this package.

## LoongSuite delta

The intentional product delta is kept small and explicit:

- `_compat.py` adapts the upstream invocation-oriented handler API to
  LoongSuite's own `opentelemetry-util-genai` implementation and
  `ExtendedTelemetryHandler`; the upstream 1.x GenAI util is not a runtime
  dependency of this package.
- `_stream.py` is the upstream GenAI stream lifecycle helper with Python 3.8
  source compatibility and LoongSuite fail-open stream ownership.
- provider source and tests avoid PEP 604 unions and PEP 585 built-in
  generics. OSS metadata remains `requires-python >=3.9` because the supported
  `google-genai>=1.32` SDK itself requires Python 3.9, while the source-level
  compatibility lets Robin normalize the packaged instrumentation wheel to
  its Python 3.8 commercial install floor without introducing import-time
  syntax or annotation failures.
- provider-owned prepare, response mapping, failure reporting, stream wrapping,
  chunk accumulation, finalization, embeddings, and automatic tool telemetry
  run through `hook_advice`; the Google SDK call, SDK stream iteration, and
  application tool callback remain outside advice and execute exactly once.
- standard OpenTelemetry suppression is honored without adding Robin's
  commercial `_SUPPRESS_LLM_SDK_KEY`;
- `execute_tool` remains an INTERNAL span through the extended handler;
- Google thought parts map to the standard `Reasoning` message part;
- upstream 1.0 event/content behavior is preserved while the repository still
  ships the 0.x shared GenAI util;
- LoongSuite completion hooks, metrics, and asynchronous multimodal processing
  remain active.

The audit also found lifecycle defects that are still present in the frozen
release and the audited upstream `main` snapshot. They are kept as isolated
provider fixes with regression tests:

- tool wrapping copies `GenerateContentConfig` so a caller can safely reuse it;
- an invalid configuration dictionary is passed through unchanged instead of
  being replaced by an empty validated configuration;
- synchronous and asynchronous stream-construction failures finish the
  invocation instead of leaking its span/context;
- async stream close supports the Google SDK's `aclose()` contract;
- generation and Interactions streams record time to first token;
- Interactions streaming accepts both the upstream test fixture's
  `interaction_completed` event and the dotted `interaction.completed` event
  emitted by the real Google GenAI 2.x API;
- embedding raw-response state is reset with a `ContextVar` token even when a
  request fails;
- explicit `OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT=false` is preserved.
- probe failures cannot replace a Google SDK result, yielded chunk, application
  tool result, cancellation, `GeneratorExit`, or original provider exception;
- stream context is detached before returning the stream, finalization is
  idempotent, and cross-thread/cross-Context consumption does not reuse the
  creation token;
- MCP tool metadata collection does not call application callbacks before the
  real provider request.

The upstream and OpenInference comparisons both provide useful partial
isolation, but neither snapshot satisfies this complete contract.
OpenInference protects many span mutation/finalization calls and falls back to
the raw stream when wrapper construction fails, while its request extraction
and stream chunk accumulator can still fail before or during business
delivery. The canonical upstream snapshot still mixes response mapping and
`invocation.stop()`/`fail()` with the provider-call `try` block. LoongSuite
keeps this lifecycle delta until an equivalent provider-neutral contract lands
upstream.

Package metadata, release naming, bootstrap registration, local live tests,
and sanitized VCR cassettes are LoongSuite-owned.

## OSS and Robin runtime boundary

The provider hooks have one lifecycle path in both distributions:

1. Google GenAI wrappers create the upstream-shaped invocation facade in
   `_compat.py`.
2. The facade delegates start, stop, failure, embedding, and tool lifecycle to
   LoongSuite's `ExtendedTelemetryHandler`.
3. The installed implementation of the shared GenAI util determines the
   telemetry extensions.

In OSS, the source dependency is named `opentelemetry-util-genai`. LoongSuite
release tooling publishes that source as `loongsuite-otel-util-genai` and
rewrites package dependencies without changing the
`opentelemetry.util.genai` import namespace. The shared util owns standard
`gen_ai.client.*` metrics, content capture, completion hooks, and multimodal
processing.

After synchronization into Robin, the same imports resolve to Robin's
commercial GenAI util. Its `EnterpriseInvocationMetricsRecorder` adds ARMS
metrics through `ExtendedTelemetryHandler`; the Google provider plugin must not
duplicate those instruments locally.

Robin also carries `_SUPPRESS_LLM_SDK_KEY` as a narrow provider overlay:

- the high-level framework handler creates its own AGENT, LLM, and TOOL spans
  and scopes the key around downstream SDK execution;
- Robin's Google GenAI `_compat._is_suppressed()` checks that private key in
  addition to standard OpenTelemetry suppression;
- suppressed SDK invocations create no Google provider LLM, EMBEDDING, or TOOL
  spans;
- context restoration is verified for success, failure, cancellation, and
  stream close so a later direct Google SDK call remains instrumented.

The private key must not be added to the OSS util or made a condition inside
the shared handler's `start_llm()`: high-level framework handlers use that same
method and must continue to create their own spans.

## Updating the baseline

For each upstream release:

1. Freeze the release tag and commit in this file.
2. Diff the provider hook modules and upstream tests against the frozen
   baseline.
3. Import upstream changes without mixing in framework instrumentation such as
   Google ADK.
4. Reapply only the imports that route invocation, handler, stream, and
   compatibility types through `_compat.py` and `_stream.py`.
5. Compare the new release with upstream `main` so unreleased fixes are tracked
   separately from LoongSuite extensions.
6. Run oldest/latest SDK tests, sanitized VCR replay, real Gemini SDK tests,
   Weaver validation, and ARMS/CMS readback before updating the pull request.

Once LoongSuite's shared GenAI util adopts the upstream 1.x invocation API,
`_compat.py` should be removed and the provider modules should return to exact
upstream imports.

## Changes already on upstream main

Do not carry these as independent LoongSuite patches. They landed upstream
after `1.0b1` and should be consumed with the next upstream release:

- interactions response ID
  ([upstream #233](https://github.com/open-telemetry/opentelemetry-python-genai/pull/233));
- interactions tool definitions and response parsing fixes
  ([upstream #271](https://github.com/open-telemetry/opentelemetry-python-genai/pull/271));
- INTERNAL span kind for `execute_tool`
  ([upstream #274](https://github.com/open-telemetry/opentelemetry-python-genai/pull/274));
- reasoning tokens included in the output-token count
  ([upstream #283](https://github.com/open-telemetry/opentelemetry-python-genai/pull/283));
- Google API error-code based `error.type`
  ([upstream #304](https://github.com/open-telemetry/opentelemetry-python-genai/pull/304)).

## Upstream contribution candidates

The following deltas are provider-independent or useful to other
instrumentation consumers and should be proposed upstream as small,
independent changes:

1. Map Google `Part(thought=True)` to the standard `Reasoning` message part
   instead of `Text`, with sync, async, and stream tests.
2. Honor OpenTelemetry's standard instrumentation suppression context in the
   invocation/handler path.
3. Respect an explicitly configured
   `OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT` value instead of unconditionally
   overwriting it during Google GenAI instrumentation.
4. Make tool wrapping non-mutating and preserve invalid configuration
   pass-through semantics.
5. Finalize invocations when sync or async stream construction raises.
6. Support SDK streams that expose `aclose()` and record TTFT on the first
   generation or Interactions stream item.
7. Parse the real Interactions SSE `interaction.completed` event while
   retaining compatibility with the underscore spelling used by existing
   fixtures.
8. Isolate the embedding raw-response `ContextVar` across failed and nested
   requests.
9. Discuss a supported handler/invocation factory injection point so vendors
   can add completion hooks, upload handling, or extra metrics without copying
   provider hooks. This needs an upstream design issue before a code change.
10. Contribute provider-neutral fail-open lifecycle patches in small steps:
    keep the provider call outside telemetry advice, return raw streams when
    wrapping fails, isolate chunk/final callbacks, detach stream context before
    ownership transfer, and preserve cancellation/exception identity.

Python 3.8 source compatibility, the 0.x-to-1.x compatibility facade,
LoongSuite-specific metrics and asynchronous multimodal processing,
release/bootstrap integration, and Robin's `_SUPPRESS_LLM_SDK_KEY` are product
or compatibility concerns and are not direct upstream contribution candidates.
