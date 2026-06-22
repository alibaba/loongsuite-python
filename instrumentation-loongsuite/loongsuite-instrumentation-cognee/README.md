# LoongSuite Cognee Instrumentation

Instrumentation for the [Cognee](https://github.com/topoteretes/cognee) AI memory
platform, producing OpenTelemetry spans following the LoongSuite gen-ai semantic
conventions.

## Span Coverage

| Span type   | Source                                                                       |
| ----------- | ---------------------------------------------------------------------------- |
| ENTRY       | wraps `cognee.add / cognify / search / recall / remember`                     |
| CHAIN       | `cognee.api.*` normalized by `CogneeAttributeSpanProcessor`                 |
| TASK        | `cognee.pipeline.task.*` normalized by `CogneeAttributeSpanProcessor`        |
| RETRIEVER   | `cognee.retrieval.*` normalized by `CogneeAttributeSpanProcessor`            |
| AGENT       | wraps `AgenticRetriever._run_tool_loop` (falls back to `get_retrieved_objects`) |
| STEP        | wraps `generate_completion` inside the ReAct loop (gated by ContextVar)       |
| TOOL        | wraps `cognee.modules.tools.execute_tool`                                    |
| EMBEDDING   | wraps non-LiteLLM `OllamaEmbeddingEngine` / `FastembedEmbeddingEngine` / `OpenAICompatibleEmbeddingEngine.embed_text` |
| LLM         | **delegated** to `loongsuite-instrumentation-litellm`                          |

The instrumentor does **not** wrap `litellm.acompletion` or
`LLMGateway.acreate_structured_output` — LLM spans (with token usage) come from
the LiteLLM instrumentor. Install both packages in production deployments:

```bash
pip install loongsuite-instrumentation-cognee loongsuite-instrumentation-litellm
```

The `cognee.llm.completion` span produced by Cognee's own tracing is rewritten
by `CogneeAttributeSpanProcessor` to `gen_ai.span.kind=CHAIN` / name
`task llm_completion`, preventing backend confusion with the LLM span below it.

## Cognee Tracing

Cognee's own tracing is disabled by default (`cognee_tracing_enabled=False`).
`CogneeInstrumentor._instrument` calls `cognee.modules.observability.enable_tracing()`
so that `cognee.*` spans are emitted on the same TracerProvider the probe set up.
If Cognee's OpenTelemetry extras are not installed, the probe logs a warning but
keeps its own ENTRY/AGENT/TOOL/STEP/EMBEDDING spans alive.

## LLM Async-Compat Wrap (Cognee v1.2.1 + instructor 1.14.5)

Cognee v1.2.1 builds its LLM client via
`instructor.from_litellm(litellm.acompletion, mode=...)`. When
`loongsuite-instrumentation-litellm` is loaded (which this package mandates),
it patches `litellm.acompletion` to an instance of
`AsyncCompletionWrapper` — a callable class with an `async def __call__`.
`inspect.iscoroutinefunction(wrapper_instance)` returns `False` for such
instances, so instructor 1.14.5's `is_async` picks the sync retry path
(`new_create_sync`), calls `litellm.acompletion(...)` *without* `await`,
and every LLM call raises
`instructor.core.exceptions.InstructorRetryException: 'coroutine' object has no attribute 'choices'`.

`CogneeInstrumentor._instrument` installs an additional wrap on
`GenericAPIAdapter.__init__` that rebuilds `self.aclient` as an explicit
`AsyncInstructor` (routing `litellm.acompletion` through a real `async def`
so instructor picks `new_create_async`). This is a pure runtime compat
fix — no span / metric / attribute is added or renamed. See
`src/opentelemetry/instrumentation/cognee/internal/_llm_compat_wrapper.py`
for the full root-cause analysis.

## Configuration

| Environment variable                                          | Default | Effect                                                                 |
| ------------------------------------------------------------- | ------- | --------------------------------------------------------------------- |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`          | `false` | Capture `gen_ai.input.messages` / `gen_ai.output.messages` / `gen_ai.tool.call.arguments` |
| `OTEL_INSTRUMENTATION_COGNEE_INTERNAL_ENABLED`               | `false` | Enable non-LiteLLM EMBEDDING wrapper (`Ollama`/`Fastembed`/`OpenAICompatible`) |
| `OTEL_INSTRUMENTATION_COGNEE_REACT_STEP_ENABLED`             | `true`  | Emit ReAct STEP spans inside the AGENT loop                            |
