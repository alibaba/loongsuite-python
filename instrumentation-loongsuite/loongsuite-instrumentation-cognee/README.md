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

## Configuration

| Environment variable                                          | Default | Effect                                                                 |
| ------------------------------------------------------------- | ------- | --------------------------------------------------------------------- |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`          | `false` | Capture `gen_ai.input.messages` / `gen_ai.output.messages` / `gen_ai.tool.call.arguments` |
| `OTEL_INSTRUMENTATION_COGNEE_INTERNAL_ENABLED`               | `false` | Enable non-LiteLLM EMBEDDING wrapper (`Ollama`/`Fastembed`/`OpenAICompatible`) |
| `OTEL_INSTRUMENTATION_COGNEE_REACT_STEP_ENABLED`             | `true`  | Emit ReAct STEP spans inside the AGENT loop                            |
