# LoongSuite LlamaIndex Instrumentation

OpenTelemetry instrumentation for [LlamaIndex](https://github.com/run-llama/llama_index)
(`llama-index-core`).

Instead of monkey-patching call sites, this package attaches to LlamaIndex's
**native instrumentation dispatcher** (`llama_index.core.instrumentation`) and
re-projects the span/event stream LlamaIndex already emits onto OpenTelemetry
spans that follow the ARMS gen-ai semantic conventions. LlamaIndex threads a
`parent_span_id` through every instrumented call, so the resulting OTel trace
preserves the logical parent/child structure (e.g. `query` → `retrieve` /
`synthesize`, `chat` → `complete`).

## Installation

```bash
pip install loongsuite-instrumentation-llama-index
```

## Usage

```python
from opentelemetry.instrumentation.llama_index import LlamaIndexInstrumentor

LlamaIndexInstrumentor().instrument()
```

## Span kinds

| LlamaIndex span                         | `gen_ai.span.kind` |
| --------------------------------------- | ------------------ |
| `*.chat` / `*.complete` / `*.predict`   | `LLM`              |
| `*.get_*_embedding*`                    | `EMBEDDING`        |
| retriever `*.retrieve`                  | `RETRIEVER`        |
| reranker / node-postprocessor           | `RERANKER`         |
| response synthesizer `*.synthesize`     | `TASK`             |
| query engine `*.query`                  | `CHAIN`            |
| chat engine / agent `*.chat` / `*.run`  | `AGENT`            |

## Content capture

Message text is captured on span attributes by default. To suppress
`gen_ai.input.messages` / `gen_ai.output.messages` while keeping the
structural spans and token metrics:

```bash
export OTEL_INSTRUMENTATION_LLAMA_INDEX_CAPTURE_CONTENT=false
```
