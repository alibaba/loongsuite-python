# LoongSuite Agno Instrumentation

This package instruments Agno 2.x agent applications with LoongSuite GenAI
semantic conventions through `opentelemetry-util-genai`.

It captures:

- agent runs as `invoke_agent` spans with `gen_ai.span.kind=AGENT`
- model calls as `chat` spans with `gen_ai.span.kind=LLM`
- Agno function calls as `execute_tool` spans with `gen_ai.span.kind=TOOL`
- token usage, prompt/response content, tool definitions, tool arguments and
  tool results according to the configured GenAI content capture mode

When `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY` is enabled,
prompt, response, and tool I/O content is written to trace span attributes.
Avoid enabling content capture for sensitive production data unless that is an
intentional observability policy.

## Installation

```shell
pip install loongsuite-distro
loongsuite-bootstrap -a install --latest
```

For local source validation:

```shell
pip install -e ./opentelemetry-instrumentation \
  -e ./util/opentelemetry-util-genai \
  -e ./instrumentation-loongsuite/loongsuite-instrumentation-agno
```

## Example

Create `demo.py`:

```python
import os

from agno.agent import Agent
from agno.models.dashscope import DashScope


def get_weather(city: str) -> str:
    return f"{city}: sunny, 24C"


agent = Agent(
    name="AgnoDashScopeDemo",
    model=DashScope(
        id=os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    ),
    tools=[get_weather],
    instructions=["When weather is requested, use the get_weather tool."],
)

response = agent.run("What is the weather in Hangzhou?")
print(response.content)

for event in agent.run("Stream a short answer.", stream=True):
    if getattr(event, "content", None):
        print(event.content, end="")
```

Collect telemetry:

```shell
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_SERVICE_NAME=loongsuite-agno-demo
export DASHSCOPE_API_KEY=YOUR_API_KEY

loongsuite-instrument python demo.py
```

The repository also includes
`examples/agno_dashscope_smoke.py`, which exercises non-streaming,
streaming, and concurrent real DashScope calls:

```shell
AGNO_SMOKE_MODE=all loongsuite-instrument \
  python instrumentation-loongsuite/loongsuite-instrumentation-agno/examples/agno_dashscope_smoke.py
```
