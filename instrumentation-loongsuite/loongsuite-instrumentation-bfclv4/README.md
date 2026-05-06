# LoongSuite BFCL v4 Instrumentation

LoongSuite Python instrumentation for the [Berkeley Function Call
Leaderboard v4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
(`bfcl-eval`, package `bfcl_eval`).

## Span Topology

```
ENTRY  enter_ai_application_system          gen_ai.span.kind=ENTRY,  op=enter
└─ AGENT  invoke_agent {test_entry_id}      gen_ai.span.kind=AGENT,  op=invoke_agent
   ├─ STEP  react step                      gen_ai.span.kind=STEP,   op=react
   │   ├─ LLM   chat {model}                (created by downstream vendor SDK probe)
   │   └─ TOOL  execute_tool {fn}           gen_ai.span.kind=TOOL,   op=execute_tool
   └─ STEP  react step
       └─ ...
```

This instrumentation deliberately does **not** create LLM spans. They are
emitted by the downstream vendor SDK probe (OpenAI / Anthropic / Google /
DashScope / LiteLLM / etc.) so that token usage and request payloads stay in
sync with the SDK that actually performed the request.

## Installation

```bash
pip install loongsuite-instrumentation-bfclv4
```

## Usage

```bash
opentelemetry-instrument bfcl generate \
    --model gpt-4o-2024-11-20-FC \
    --test-category simple_python \
    --num-threads 2
```

Or programmatically:

```python
from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

BFCLv4Instrumentor().instrument()
# ... run BFCL ...
BFCLv4Instrumentor().uninstrument()
```

## Compatibility With Downstream LLM SDK Probes

| Scenario | Recommended downstream probe |
| --- | --- |
| OpenAI / OpenAI Responses / OSS via vLLM / SGLang / DeepSeek (OpenAI-compatible) | `opentelemetry-instrumentation-openai` |
| Anthropic / Claude | `loongsuite-instrumentation-claude-agent-sdk` |
| Gemini / Google | `loongsuite-instrumentation-google-adk` |
| Qwen / DashScope | `loongsuite-instrumentation-dashscope` |
| LiteLLM | `loongsuite-instrumentation-litellm` |

## OSS Provider Notes

For OSS handlers (vLLM / SGLang served via the OpenAI-compatible API), the
BFCL probe sets `gen_ai.provider.name` to `vllm` / `sglang` / `oss` and adds
`bfcl.oss.backend` for disambiguation. Downstream OpenAI probes will still
report `gen_ai.provider.name=openai` on the LLM span; this is expected.

## Custom Attributes

| Attribute | Where | Description |
| --- | --- | --- |
| `gen_ai.framework` = `bfclv4` | ENTRY/AGENT/STEP/TOOL | Framework tag |
| `bfcl.test_category` | ENTRY/AGENT | Test category |
| `bfcl.num_threads` | ENTRY | Configured thread pool size |
| `bfcl.test_case_count` | ENTRY | Number of test cases |
| `bfcl.run_ids` | ENTRY | Whether the run targeted specific IDs |
| `bfcl.test_entry_id` | AGENT | Test entry id |
| `bfcl.turn_idx` | STEP | Multi-turn turn index (0-based) |
| `bfcl.query_mode` | STEP | `FC` or `prompting` |
| `bfcl.oss.backend` | AGENT/STEP | `vllm` / `sglang` / `unknown` (only OSS) |
| `bfcl.tool.duration_is_estimated` | TOOL | True (latency is averaged across batch) |
