# LoongSuite Microsoft AutoGen Instrumentation

This package instruments Microsoft AutoGen 0.7.x AgentChat flows.

It follows the AutoGen 0.7 native telemetry boundary:

- AutoGen core already emits native GenAI-style spans for `invoke_agent`,
  `create_agent`, and `execute_tool`.
- This instrumentation registers a span processor that converts those native
  spans to LoongSuite GenAI attributes by setting `gen_ai.span.kind`, replacing
  `gen_ai.system=autogen` with `gen_ai.provider.name=autogen`, and normalizing
  span kinds.
- AgentChat `AssistantAgent.on_messages_stream` is wrapped with
  `ExtendedTelemetryHandler.invoke_agent()` when no native AutoGen agent span is
  already active.
- AgentChat `AssistantAgent._call_llm` is wrapped with
  `ExtendedTelemetryHandler.llm()` so model calls produce LoongSuite LLM spans
  for any AutoGen `ChatCompletionClient`.

## Usage

```python
from opentelemetry.instrumentation.autogen import AutoGenInstrumentor

AutoGenInstrumentor().instrument()
```

Environment switches:

- `ARMS_AUTOGEN_INSTRUMENTATION_ENABLED=false` disables the instrumentation.
- `ARMS_AUTOGEN_AGENT_SPAN_ENABLED=false` disables the AgentChat agent wrapper.
- `ARMS_AUTOGEN_LLM_SPAN_ENABLED=false` disables the AgentChat LLM wrapper.
- `ARMS_AUTOGEN_NATIVE_SPAN_PROCESSOR_ENABLED=false` disables native span
  normalization.

The package uses `opentelemetry-util-genai` and respects the shared GenAI
content-capture environment variables.
