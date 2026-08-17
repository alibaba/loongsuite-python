# LoongSuite DSPy Instrumentation

OpenTelemetry GenAI instrumentation for [DSPy](https://github.com/stanfordnlp/dspy) (`dspy >= 3.0.0`).

## What it produces

This instrumentation emits the **framework layer** of a DSPy trace:

| Span kind | Source |
| --- | --- |
| `ENTRY` | the outermost DSPy module call |
| `AGENT` | `dspy.ReAct` (and subclasses such as `dspy.CodeAct`) |
| `STEP` | one ReAct Reasoning-Acting round |
| `CHAIN` | every other module (`Predict`, `ChainOfThought`, custom `dspy.Module`) and `dspy.Evaluate` |
| `TOOL` | `dspy.Tool` invocation |
| `RETRIEVER` | `dspy.Retrieve` (and subclasses such as `DatabricksRM`, `WeaviateRM`) |

## Required companion instrumentation

> **`loongsuite-instrumentation-litellm` must be enabled in the same process.**

DSPy routes every built-in model call (`LM.forward` / `aforward`, all `model_type`
variants) and every `dspy.Embedder` call through LiteLLM. `LLM` / `EMBEDDING`
spans and all token usage metrics therefore come from
`loongsuite-instrumentation-litellm`, and this instrumentation deliberately does
not emit them — doing so would duplicate spans and double-count tokens.

Without the LiteLLM instrumentation a trace contains only the framework
skeleton: no `LLM` spans and no token usage.

Recommended in addition: `opentelemetry-instrumentation-openai-v2`, which covers
user-defined `dspy.BaseLM` subclasses that call the OpenAI SDK directly and so
bypass LiteLLM.

## Usage

```python
from opentelemetry.instrumentation.dspy import DSPyInstrumentor
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor

DSPyInstrumentor().instrument()
LiteLLMInstrumentor().instrument()

# ... use DSPy as normal ...

DSPyInstrumentor().uninstrument()
```

Under `loongsuite-instrument` / `opentelemetry-instrument` both instrumentations
are loaded automatically through their entry points.

## Configuration

Message content (prompts, module inputs/outputs, tool arguments and results,
retrieved documents) follows the shared `util-genai` switches:

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```

DSPy-specific switches:

| Environment variable | Default | Effect |
| --- | --- | --- |
| `OTEL_INSTRUMENTATION_DSPY_CAPTURE_ENTRY_SPAN` | `true` | Wrap the outermost module call in an `ENTRY` span. |
| `OTEL_INSTRUMENTATION_DSPY_CAPTURE_MODEL_NAME` | `true` | Best-effort `gen_ai.request.model` on framework spans, read from `dspy.settings.lm.model` and normalized exactly like the LiteLLM instrumentation normalizes it. Set to `false` to leave it unset. |
| `OTEL_INSTRUMENTATION_DSPY_REACT_STEP_ENABLED` | `true` | Emit `STEP` spans for ReAct rounds. |
| `OTEL_INSTRUMENTATION_DSPY_ROOT_SAMPLE_RATIO` | `1.0` | Fraction of outermost DSPy calls that produce framework spans. Optimizer compilation (`teleprompt/*`) replays a program many times; lowering this bounds the span volume. The decision is taken once per outermost call and covers its whole subtree, so a dropped program never emits a partial tree. It does **not** drop the LiteLLM `LLM` spans below it — combine it with a trace-level OTel sampler when whole traces should be dropped. |

## Notes and limitations

* **Optimizers** (`dspy.teleprompt.*`) are not instrumented directly; they are
  observed through the module and tool callbacks they trigger, which multiplies
  span volume by the number of optimization trials. See
  `OTEL_INSTRUMENTATION_DSPY_ROOT_SAMPLE_RATIO` above.
* **Retrieved documents** carry ordinal ids and no relevance score: DSPy
  retrievers return bare passages. `score` is left unset rather than
  synthesized.
* **Aggregate token usage** from `Prediction.get_lm_usage()` is recorded on the
  `AGENT` span as `gen_ai.usage.*` attributes only (and only when
  `dspy.settings.track_usage` is enabled). It never reaches token metrics — the
  LiteLLM instrumentation is the single source of token measurements.
* **DSPy cache hits** produce framework spans but no `LLM` span and no tokens,
  because no model call is made. This is expected, not a gap.
* **Threaded execution** (`dspy.Parallel`, `Evaluate(num_threads=...)`) runs
  worker threads with a fresh OpenTelemetry context, so spans created in those
  threads do not nest under the caller's span.
