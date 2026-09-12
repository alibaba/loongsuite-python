# LoongSuite LangChain Instrumentation

This package provides LoongSuite instrumentation for LangChain applications, allowing you to automatically trace and monitor your LangChain workflows. For details on usage and installation of LoongSuite and Jaeger, please refer to [LoongSuite Documentation](https://github.com/alibaba/loongsuite-python/blob/main/README.md).

## Installation

### Install instrumentation (recommended: root README Option C)

```bash
# Step 1: install LoongSuite distro
pip install loongsuite-distro

# Step 2 (Option C): install this instrumentation from PyPI
pip install loongsuite-instrumentation-langchain
```

## RUN

### Build the Example

Follow the official [LangChain Documentation](https://python.langchain.com/docs/introduction/) to create a sample file named `demo.py`.

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os


chatLLM = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    temperature=0,
    stream_usage=True,
)
messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    ),
]
res = chatLLM.invoke(messages)
print(res)
```

## Quick Start

You can automatically instrument your LangChain application using the `loongsuite-instrument` command:

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
loongsuite-instrument \
    --traces_exporter console \
    --metrics_exporter console \
    --logs_exporter none \
    python your_langchain_app.py
```
If everything is working correctly, you should see logs similar to the following
```json
{
    "name": "chat qwen-plus",
    "context": {
        "trace_id": "0x153d9f32aeaef815a7ddc9ec406ef8fc",
        "span_id": "0xc0c4107603054139",
        "trace_state": "[]"
    },
    "kind": "SpanKind.CLIENT",
    "parent_id": null,
    "start_time": "2026-03-10T06:04:56.411044Z",
    "end_time": "2026-03-10T06:04:57.205725Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {
        "gen_ai.operation.name": "chat",
        "gen_ai.span.kind": "LLM",
        "gen_ai.request.model": "qwen-plus",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.temperature": 0.0,
        "gen_ai.response.finish_reasons": [
            "stop"
        ],
        "gen_ai.response.model": "qwen-plus",
        "gen_ai.usage.input_tokens": 36,
        "gen_ai.usage.output_tokens": 8,
        "gen_ai.usage.total_tokens": 44,
        "gen_ai.input.messages": "[{\"role\":\"system\",\"parts\":[{\"content\":\"You are a helpful assistant that translates English to French.\",\"type\":\"text\"}]},{\"role\":\"user\",\"parts\":[{\"content\":\"Translate this sentence from English to French. I love programming.\",\"type\":\"text\"}]}]",
        "gen_ai.output.messages": "[{\"role\":\"assistant\",\"parts\":[{\"content\":\"J\u2019adore la programmation.\",\"type\":\"text\"}],\"finish_reason\":\"stop\"}]"
    },
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.40.0",
            "telemetry.auto.version": "0.61b0",
            "service.name": "unknown_service"
        },
        "schema_url": ""
    }
}

```

## Forwarding OTLP Data to the Backend
```shell
export OTEL_SERVICE_NAME=<service_name>
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=<trace_endpoint>
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=<metrics_endpoint>

export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY

loongsuite-instrument <your_run_command>

```


## Traced Operations

| Operation | Span Kind | Attributes |
|-----------|-----------|------------|
| Chain | `CHAIN` | `gen_ai.operation.name=chain`, `gen_ai.span.kind=CHAIN`, `input.value`, `output.value` (when content capture enabled). Span name: `chain {run.name}` (e.g. RetrievalQA, StuffDocumentsChain, LLMChain) |
| LLM / Chat | `LLM` | `gen_ai.operation.name=chat`, `gen_ai.request.model`, token usage |
| Agent | `AGENT` | `gen_ai.operation.name=invoke_agent` |
| ReAct Step | `STEP` | `gen_ai.operation.name=react`, `gen_ai.react.round`, `gen_ai.react.finish_reason` |
| Tool | `TOOL` | `gen_ai.operation.name=execute_tool` |
| Retriever | `RETRIEVER` | `gen_ai.operation.name=retrieval` |
| Reranker | `RERANKER` | `gen_ai.operation.name=rerank_documents`, `gen_ai.request.model`, `gen_ai.rerank.documents.count`, `gen_ai.request.top_k`, `gen_ai.rerank.input_documents`, `gen_ai.rerank.output_documents` (when content capture enabled) |

ReAct Step spans are created for each Reasoning-Acting iteration, with the hierarchy: Agent > ReAct Step > LLM/Tool. Supported agent types:

- **AgentExecutor** (LangChain 0.x / 1.x) — detected by `run.name`
- **LangGraph `create_react_agent`** — detected by `Run.metadata` (requires
  `loongsuite-instrumentation-langgraph`). When invoked inside an outer graph
  node, the agent span inherits the node's name for better readability.

## Requirements

- Python >= 3.9
- LangChain >= 0.1.0
- OpenTelemetry >= 1.20.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This instrumentation was inspired by and builds upon the excellent work done by the [OpenInference](https://github.com/Arize-ai/openinference) project. We acknowledge their contributions to the OpenTelemetry instrumentation ecosystem for AI/ML frameworks.

## License

This project is licensed under the Apache License 2.0.