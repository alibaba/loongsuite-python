# LoongSuite Agent-Reach Instrumentation

OpenTelemetry instrumentation for [Agent-Reach](https://github.com/Panniantong/Agent-Reach)
(`pip install agent-reach`).

## Coverage

| Span | Operation | Patch point |
|------|-----------|-------------|
| ENTRY | `enter` | `agent_reach.cli.main` |
| TOOL | `execute_tool` | `agent_reach.doctor.check_all` |
| TOOL | `execute_tool` | every `Channel.check` override (dynamic discovery via `register_post_import_hook`) |
| TOOL | `execute_tool` | `agent_reach.probe.probe_command` |
| TOOL | `execute_tool` | `agent_reach.transcribe.transcribe` |
| TOOL | `execute_tool` | `agent_reach.transcribe.transcribe_chunk` |
| TOOL | `execute_tool` | `agent_reach.integrations.mcp_server.create_server` (only when `loongsuite-instrumentation-mcp` is NOT active) |

Span hierarchy:

```
ENTRY (enter_ai_application_system)
  ├─ TOOL (execute_tool doctor)
  │    ├─ TOOL (execute_tool channel-github)
  │    │    └─ TOOL (execute_tool probe-gh)
  │    ├─ TOOL (execute_tool channel-twitter)
  │    │    └─ TOOL (execute_tool probe-twitter)
  │    └─ … (remaining channels)
  ├─ TOOL (execute_tool agent-reach-transcribe)
  │    └─ TOOL (execute_tool whisper-transcribe) × N
  └─ TOOL (execute_tool mcp-get_status) [conditional]
```

Sensitive content (`gen_ai.tool.call.arguments` / `gen_ai.tool.call.result`)
is gated behind `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`. When
the flag is unset / `NO_CONTENT`, only non-PII diagnostic attributes
(`agent_reach.channel.status`, `agent_reach.probe.status`, etc.) are emitted.

## Usage

```python
from opentelemetry.instrumentation.agent_reach import AgentReachInstrumentor

AgentReachInstrumentor().instrument()
```
