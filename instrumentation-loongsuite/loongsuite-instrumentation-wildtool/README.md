# LoongSuite WildToolBench Instrumentation

OpenTelemetry instrumentation for the [WildToolBench](https://github.com/yupeijei1997/WildToolBench) benchmark framework.

## Installation

WildToolBench is not available on PyPI. Install it from source:

```bash
pip install -e /path/to/WildToolBench/wild-tool-bench
pip install loongsuite-instrumentation-wildtool
```

## Requirements

- **OpenAI provider instrumentation**: To produce LLM spans, you must also enable an OpenAI provider instrumentation (e.g., `opentelemetry-instrumentation-openai` or LoongSuite's equivalent). This plugin creates ENTRY/AGENT/CHAIN/STEP/TOOL spans but does **not** create LLM spans itself.

## Usage

```python
from opentelemetry.instrumentation.wildtool import WildToolInstrumentor

WildToolInstrumentor().instrument()

# Run WildToolBench as usual — spans are automatically generated.
```

## Span Topology

```
ENTRY (enter_ai_application_system)
└── AGENT (invoke_agent wildtool)
    └── CHAIN (workflow task_{idx})
        └── STEP (react step)
            ├── [LLM span — provider instrumentation]
            └── TOOL (execute_tool {tool_name})
```

## Patch Points

| # | Target | Span Type |
|---|--------|-----------|
| P1 | `multi_threaded_inference` | ENTRY |
| P2 | `BaseHandler.inference_multi_turn` | AGENT |
| P3 | `BaseHandler.inference_and_eval_multi_step` | CHAIN + TOOL |
| P4 | `BaseHandler._request_tool_call` | STEP |
| P5 | `BaseHandler._parse_api_response` | (token extraction) |

## Round 2 fixes (see `llm-dev/execute.md` § "修订记录 (Round 2 fix)")

- **H1**: TOOL span is now parented on STEP, not CHAIN. Strategy A enhanced — the chain wrapper holds a `round → STEP span` map and uses `trace.set_span_in_context(step_span)` to anchor each post-hoc TOOL span on the matching STEP. STEP `SpanContext`s remain valid parents even after `end()`.
- **H2 (provider-name fallback)**: `opentelemetry-instrumentation-openai-v2 == 0.62b1` only emits the legacy `gen_ai.system` attribute on its LLM span; the new `gen_ai.provider.name` attribute is missing. As a *pure fallback* the wildtool plugin writes both `gen_ai.system="openai"` and `gen_ai.provider.name="openai"` on the **STEP** span (not on the LLM span — that is owned by the OpenAI v2 instrumentation and we do **not** patch it). Once the OpenAI v2 instrumentation upstream emits `gen_ai.provider.name` natively this fallback can be removed.
- **M1**: CHAIN span now carries `input.value` (last user message in `inference_data["messages"]`, truncated to 4096 chars) and `output.value` (JSON of `action_name_label`/`task_idx`/`is_optimal`).
- **M2**: STEP span now carries `gen_ai.react.finish_reason` on error paths. Mapping table is in `execute.md` § "M2: gen_ai.react.finish_reason 取值映射".
- **M3**: TOOL span explicitly writes `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` / `gen_ai.tool.description`, bypassing `OTEL_INSTRUMENTATION_GENAI_CAPTURE_*` gating in `opentelemetry-util-genai`. The custom `wildtool.tool.execution_mode = "ground_truth_replay"` is preserved.
