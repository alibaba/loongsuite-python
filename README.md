# loongsuite-python-agent
<img src="docs/_assets/img/loongsuite-logo.png" width="600" height="100">

<div align="center">

**English** | [简体中文](README-zh.md)

</div>

## Introduction
Loongsuite Python Agent is a key component of LoongSuite, Alibaba's unified observability data collection suite, providing instrumentation for Python applications. 

LoongSuite includes the following key components:
* [LoongCollector](https://github.com/alibaba/loongcollector): universal node agent, which prodivdes log collection, prometheus metric collection, and network and security collection capabilities based on eBPF.
* [LoongSuite Python Agent](https://github.com/alibaba/loongsuite-python-agent): a process agent providing instrumentation for python applications.
* [LoongSuite Go Agent](https://github.com/alibaba/loongsuite-go-agent): a process agent for golang with compile time instrumentation.
* [LoongSuite Java Agent](https://github.com/alibaba/loongsuite-java-agent): a process agent for Java applications.
* Other upcoming language agent.

Loongsuite Python Agent is also a customized distribution of upstream [OTel Python Agent](https://github.com/open-telemetry/opentelemetry-python-contrib), with enhanced support for popular AI agent framework. 
The implementation follows the latest GenAI [semantic conventions](https://github.com/open-telemetry/semantic-conventions).

## Supported frameworks and components

<a id="supported-frameworks-and-components"></a>

### LoongSuite instrumentation

Source tree: [`instrumentation-loongsuite/`](instrumentation-loongsuite).

| Framework/Components | Docs | Release |
|--------|------|---------|
| [AgentScope](https://github.com/agentscope-ai/agentscope) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-agentscope/README.md) | [PyPI](https://pypi.org/project/loongsuite-instrumentation-agentscope/) |
| [Agno](https://github.com/agno-agi/agno) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-agno/README.md) | in dev |
| [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-anthropic/README.rst) | in dev |
| [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-claude-agent-sdk/README.md) | [PyPI](https://pypi.org/project/loongsuite-instrumentation-claude-agent-sdk/) |
| [QwenPaw](https://github.com/agentscope-ai/QwenPaw) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-qwenpaw/README.md) | [PyPI](https://pypi.org/project/loongsuite-instrumentation-qwenpaw/) |
| [CrewAI](https://github.com/crewAIInc/crewAI) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-crewai/README.md) | [PyPI](https://pypi.org/project/loongsuite-instrumentation-crewai/) |
| [DashScope](https://github.com/dashscope/dashscope-sdk-python) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-dashscope/README.md) | [PyPI](https://pypi.org/project/loongsuite-instrumentation-dashscope/) |
| [Dify](https://github.com/langgenius/dify) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-dify/README.md) | in dev |
| [Google ADK](https://github.com/google/adk-python) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-google-adk/README.md) | [PyPI](https://pypi.org/project/loongsuite-instrumentation-google-adk/) |
| [LangChain](https://github.com/langchain-ai/langchain) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-langchain/README.md) | [PyPI](https://pypi.org/project/loongsuite-instrumentation-langchain/) |
| [LangGraph](https://github.com/langchain-ai/langgraph) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-langgraph/README.md) | [PyPI](https://pypi.org/project/loongsuite-instrumentation-langgraph/) |
| [LiteLLM](https://github.com/BerriAI/litellm) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-litellm/README.md) | [PyPI](https://pypi.org/project/loongsuite-instrumentation-litellm/) |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-mcp/README.md) | in dev |
| [Mem0](https://github.com/mem0ai/mem0) | [GUIDE](instrumentation-loongsuite/loongsuite-instrumentation-mem0/README.md) | [PyPI](https://pypi.org/project/loongsuite-instrumentation-mem0/) |

**Distro and helpers:**

- **loongsuite-distro** — [https://pypi.org/project/loongsuite-distro/](https://pypi.org/project/loongsuite-distro/) (`loongsuite-instrument`, `loongsuite-bootstrap`)
- **loongsuite-util-genai** — [https://pypi.org/project/loongsuite-util-genai/](https://pypi.org/project/loongsuite-util-genai/)
- **loongsuite-site-bootstrap** — [https://pypi.org/project/loongsuite-site-bootstrap/](https://pypi.org/project/loongsuite-site-bootstrap/).

### OpenTelemetry instrumentation — generative workloads

Source tree: [`instrumentation-genai/`](instrumentation-genai). These distributions follow OpenTelemetry **generative** semantic conventions (`opentelemetry-instrumentation-*` on PyPI).

| Framework/Components | Docs | Release |
|--------|------|---------|
| [Anthropic](https://github.com/anthropics/anthropic-sdk-python) | [GUIDE](instrumentation-genai/opentelemetry-instrumentation-anthropic/README.rst) | [PyPI](https://pypi.org/project/opentelemetry-instrumentation-anthropic/) |
| [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) | [GUIDE](instrumentation-genai/opentelemetry-instrumentation-claude-agent-sdk/README.rst) | [PyPI](https://pypi.org/project/opentelemetry-instrumentation-claude-agent-sdk/) |
| [Google GenAI](https://github.com/googleapis/python-genai) | [GUIDE](instrumentation-genai/opentelemetry-instrumentation-google-genai/README.rst) | [PyPI](https://pypi.org/project/opentelemetry-instrumentation-google-genai/) |
| [LangChain](https://github.com/langchain-ai/langchain) | [GUIDE](instrumentation-genai/opentelemetry-instrumentation-langchain/README.rst) | [PyPI](https://pypi.org/project/opentelemetry-instrumentation-langchain/) |
| [OpenAI Agents](https://github.com/openai/openai-agents-python) | [GUIDE](instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/README.rst) | [PyPI](https://pypi.org/project/opentelemetry-instrumentation-openai-agents-v2/) |
| [OpenAI](https://github.com/openai/openai-python) | [GUIDE](instrumentation-genai/opentelemetry-instrumentation-openai-v2/README.rst) | [PyPI](https://pypi.org/project/opentelemetry-instrumentation-openai-v2/) |
| [Vertex AI](https://github.com/googleapis/python-aiplatform) | [GUIDE](instrumentation-genai/opentelemetry-instrumentation-vertexai/README.rst) | [PyPI](https://pypi.org/project/opentelemetry-instrumentation-vertexai/) |
| [Weaviate](https://github.com/weaviate/weaviate) | [GUIDE](instrumentation-genai/opentelemetry-instrumentation-weaviate/README.rst) | [PyPI](https://pypi.org/project/opentelemetry-instrumentation-weaviate/) |

> **Note:** With LoongSuite’s distro, install these together with [**loongsuite-distro**](https://pypi.org/project/loongsuite-distro/) and **`loongsuite-bootstrap`** / [**loongsuite-util-genai**](https://pypi.org/project/loongsuite-util-genai/). Avoid mixing [**loongsuite-util-genai**](https://pypi.org/project/loongsuite-util-genai/) with the community **opentelemetry-util-genai** (see [manual `pip` installs](#install-step-2-options)).

### OpenTelemetry instrumentation

Source tree: [`instrumentation/`](instrumentation). General application and library instrumentations; PyPI project is always `https://pypi.org/project/opentelemetry-instrumentation-<name>/`. Each line below links to that URL and the package README in this repo.

<details>
<summary><b>All <code>instrumentation/</code> packages (click to expand)</b></summary>

- **opentelemetry-instrumentation-aio-pika** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-aio-pika/), [readme](instrumentation/opentelemetry-instrumentation-aio-pika/README.rst)
- **opentelemetry-instrumentation-aiohttp-client** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-aiohttp-client/), [readme](instrumentation/opentelemetry-instrumentation-aiohttp-client/README.rst)
- **opentelemetry-instrumentation-aiohttp-server** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-aiohttp-server/), [readme](instrumentation/opentelemetry-instrumentation-aiohttp-server/README.rst)
- **opentelemetry-instrumentation-aiokafka** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-aiokafka/), [readme](instrumentation/opentelemetry-instrumentation-aiokafka/README.rst)
- **opentelemetry-instrumentation-aiopg** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-aiopg/), [readme](instrumentation/opentelemetry-instrumentation-aiopg/README.rst)
- **opentelemetry-instrumentation-asgi** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-asgi/), [readme](instrumentation/opentelemetry-instrumentation-asgi/README.rst)
- **opentelemetry-instrumentation-asyncclick** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-asyncclick/), [readme](instrumentation/opentelemetry-instrumentation-asyncclick/README.rst)
- **opentelemetry-instrumentation-asyncio** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-asyncio/), [readme](instrumentation/opentelemetry-instrumentation-asyncio/README.rst)
- **opentelemetry-instrumentation-asyncpg** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-asyncpg/), [readme](instrumentation/opentelemetry-instrumentation-asyncpg/README.rst)
- **opentelemetry-instrumentation-aws-lambda** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-aws-lambda/), [readme](instrumentation/opentelemetry-instrumentation-aws-lambda/README.rst)
- **opentelemetry-instrumentation-boto** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-boto/), [readme](instrumentation/opentelemetry-instrumentation-boto/README.rst)
- **opentelemetry-instrumentation-boto3sqs** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-boto3sqs/), [readme](instrumentation/opentelemetry-instrumentation-boto3sqs/README.rst)
- **opentelemetry-instrumentation-botocore** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-botocore/), [readme](instrumentation/opentelemetry-instrumentation-botocore/README.rst)
- **opentelemetry-instrumentation-cassandra** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-cassandra/), [readme](instrumentation/opentelemetry-instrumentation-cassandra/README.rst)
- **opentelemetry-instrumentation-celery** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-celery/), [readme](instrumentation/opentelemetry-instrumentation-celery/README.rst)
- **opentelemetry-instrumentation-click** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-click/), [readme](instrumentation/opentelemetry-instrumentation-click/README.rst)
- **opentelemetry-instrumentation-confluent-kafka** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-confluent-kafka/), [readme](instrumentation/opentelemetry-instrumentation-confluent-kafka/README.rst)
- **opentelemetry-instrumentation-dbapi** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-dbapi/), [readme](instrumentation/opentelemetry-instrumentation-dbapi/README.rst)
- **opentelemetry-instrumentation-django** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-django/), [readme](instrumentation/opentelemetry-instrumentation-django/README.rst)
- **opentelemetry-instrumentation-elasticsearch** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-elasticsearch/), [readme](instrumentation/opentelemetry-instrumentation-elasticsearch/README.rst)
- **opentelemetry-instrumentation-falcon** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-falcon/), [readme](instrumentation/opentelemetry-instrumentation-falcon/README.rst)
- **opentelemetry-instrumentation-fastapi** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-fastapi/), [readme](instrumentation/opentelemetry-instrumentation-fastapi/README.rst)
- **opentelemetry-instrumentation-flask** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-flask/), [readme](instrumentation/opentelemetry-instrumentation-flask/README.rst)
- **opentelemetry-instrumentation-grpc** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-grpc/), [readme](instrumentation/opentelemetry-instrumentation-grpc/README.rst)
- **opentelemetry-instrumentation-httpx** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-httpx/), [readme](instrumentation/opentelemetry-instrumentation-httpx/README.rst)
- **opentelemetry-instrumentation-jinja2** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-jinja2/), [readme](instrumentation/opentelemetry-instrumentation-jinja2/README.rst)
- **opentelemetry-instrumentation-kafka-python** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-kafka-python/), [readme](instrumentation/opentelemetry-instrumentation-kafka-python/README.rst)
- **opentelemetry-instrumentation-logging** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-logging/), [readme](instrumentation/opentelemetry-instrumentation-logging/README.rst)
- **opentelemetry-instrumentation-mysql** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-mysql/), [readme](instrumentation/opentelemetry-instrumentation-mysql/README.rst)
- **opentelemetry-instrumentation-mysqlclient** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-mysqlclient/), [readme](instrumentation/opentelemetry-instrumentation-mysqlclient/README.rst)
- **opentelemetry-instrumentation-pika** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-pika/), [readme](instrumentation/opentelemetry-instrumentation-pika/README.rst)
- **opentelemetry-instrumentation-psycopg** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-psycopg/), [readme](instrumentation/opentelemetry-instrumentation-psycopg/README.rst)
- **opentelemetry-instrumentation-psycopg2** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-psycopg2/), [readme](instrumentation/opentelemetry-instrumentation-psycopg2/README.rst)
- **opentelemetry-instrumentation-pymemcache** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-pymemcache/), [readme](instrumentation/opentelemetry-instrumentation-pymemcache/README.rst)
- **opentelemetry-instrumentation-pymongo** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-pymongo/), [readme](instrumentation/opentelemetry-instrumentation-pymongo/README.rst)
- **opentelemetry-instrumentation-pymssql** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-pymssql/), [readme](instrumentation/opentelemetry-instrumentation-pymssql/README.rst)
- **opentelemetry-instrumentation-pymysql** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-pymysql/), [readme](instrumentation/opentelemetry-instrumentation-pymysql/README.rst)
- **opentelemetry-instrumentation-pyramid** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-pyramid/), [readme](instrumentation/opentelemetry-instrumentation-pyramid/README.rst)
- **opentelemetry-instrumentation-redis** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-redis/), [readme](instrumentation/opentelemetry-instrumentation-redis/README.rst)
- **opentelemetry-instrumentation-remoulade** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-remoulade/), [readme](instrumentation/opentelemetry-instrumentation-remoulade/README.rst)
- **opentelemetry-instrumentation-requests** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-requests/), [readme](instrumentation/opentelemetry-instrumentation-requests/README.rst)
- **opentelemetry-instrumentation-sqlalchemy** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-sqlalchemy/), [readme](instrumentation/opentelemetry-instrumentation-sqlalchemy/README.rst)
- **opentelemetry-instrumentation-sqlite3** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-sqlite3/), [readme](instrumentation/opentelemetry-instrumentation-sqlite3/README.rst)
- **opentelemetry-instrumentation-starlette** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-starlette/), [readme](instrumentation/opentelemetry-instrumentation-starlette/README.rst)
- **opentelemetry-instrumentation-system-metrics** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-system-metrics/), [readme](instrumentation/opentelemetry-instrumentation-system-metrics/README.rst)
- **opentelemetry-instrumentation-threading** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-threading/), [readme](instrumentation/opentelemetry-instrumentation-threading/README.rst)
- **opentelemetry-instrumentation-tornado** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-tornado/), [readme](instrumentation/opentelemetry-instrumentation-tornado/README.rst)
- **opentelemetry-instrumentation-tortoiseorm** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-tortoiseorm/), [readme](instrumentation/opentelemetry-instrumentation-tortoiseorm/README.rst)
- **opentelemetry-instrumentation-urllib** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-urllib/), [readme](instrumentation/opentelemetry-instrumentation-urllib/README.rst)
- **opentelemetry-instrumentation-urllib3** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-urllib3/), [readme](instrumentation/opentelemetry-instrumentation-urllib3/README.rst)
- **opentelemetry-instrumentation-wsgi** — [PyPI](https://pypi.org/project/opentelemetry-instrumentation-wsgi/), [readme](instrumentation/opentelemetry-instrumentation-wsgi/README.rst)

</details>

## Quick start

This walkthrough uses **[AgentScope](https://github.com/agentscope-ai/agentscope)**. The same exporter and `loongsuite-instrument` patterns apply to other stacks once their instrumentations are installed.

### Prepare a demo (AgentScope ReAct example)

Use the upstream **[ReAct agent example](https://github.com/agentscope-ai/agentscope/tree/main/examples/agent/react_agent)** as reference: you can clone AgentScope or align with that folder’s `main.py`.

**Step 1 — Install AgentScope**

  ```bash
  pip install agentscope
  ```

**Step 2 - Configure DashScope**

  ```bash
  export DASHSCOPE_API_KEY={your_api_key}
  ```

  Replace `{your_api_key}` with a valid key from the [DashScope console](https://bailian.console.aliyun.com/#/api-key).

  To connect to a different model API instead of DashScope, see the AgentScope documentation: [Model tutorial](https://doc.agentscope.io/tutorial/task_model.html).

**Step 3 - Create ReAct Agent**

  ```python
  # -*- coding: utf-8 -*-
  """The main entry point of the ReAct agent example."""
  import asyncio
  import os

  from agentscope.agent import ReActAgent, UserAgent
  from agentscope.formatter import DashScopeChatFormatter
  from agentscope.memory import InMemoryMemory
  from agentscope.model import DashScopeChatModel
  from agentscope.tool import (
      Toolkit,
      execute_shell_command,
      execute_python_code,
      view_text_file,
  )


  async def main() -> None:
      """The main entry point for the ReAct agent example."""
      toolkit = Toolkit()

      toolkit.register_tool_function(execute_shell_command)
      toolkit.register_tool_function(execute_python_code)
      toolkit.register_tool_function(view_text_file)

      agent = ReActAgent(
          name="Friday",
          sys_prompt="You are a helpful assistant named Friday.",
          model=DashScopeChatModel(
              api_key=os.environ.get("DASHSCOPE_API_KEY"),
              model_name="qwen-max",
              enable_thinking=False,
              stream=True,
          ),
          formatter=DashScopeChatFormatter(),
          toolkit=toolkit,
          memory=InMemoryMemory(),
      )

      user = UserAgent("User")

      msg = None
      while True:
          msg = await user(msg)
          if msg.get_text_content() == "exit":
              break
          msg = await agent(msg)


  asyncio.run(main())
  ```

### Install and run loongsuite

<a id="install-and-run-loongsuite"></a>

Recommended integration approach: **automatic instrumentation** with **`loongsuite-instrument`** after installing **`loongsuite-distro`** and your instrumentations (via **`loongsuite-bootstrap`** or manual `pip`).

**Step 1 — Install the distro**

  ```bash
  pip install loongsuite-distro
  ```

  Optional: `pip install loongsuite-distro[otlp]` for OTLP extras ([loongsuite-distro README](loongsuite-distro/README.rst)).

**Step 2 — Install instrumentations**

  Use **`loongsuite-bootstrap`** (shipped with `loongsuite-distro`) to install LoongSuite wheels from a [GitHub Release](https://github.com/alibaba/loongsuite-python-agent/releases) tarball and compatible `opentelemetry-instrumentation-*` versions from PyPI. Bootstrap performs a **two-phase** install: LoongSuite artifacts from the release, then pinned OpenTelemetry instrumentation packages (see [docs/loongsuite-release.md](docs/loongsuite-release.md)).

  Pick **one** of the following:

  <a id="install-step-2-options"></a>

- **Option A — Install everything** from a release:

  ```bash
  loongsuite-bootstrap -a install --latest
  # for specific version: loongsuite-bootstrap -a install --version X.Y.Z
  ```

- **Option B — Auto-detect** (lean environments): install only instrumentations for libraries already present:

  ```bash
  loongsuite-bootstrap -a install --latest --auto-detect
  ```

- **Option C — Manual `pip`**: install packages yourself from PyPI using the names in [Supported frameworks and components](#supported-frameworks-and-components).

  ```bash
  pip install loongsuite-instrumentation-agentscope
  ```

  > **Note:** If you need packages under [`instrumentation-genai/`](instrumentation-genai), use **Option A or B** together with **`loongsuite-distro`** / **`loongsuite-bootstrap`**. Relying only on manual `pip` can cause **dependency resolution conflicts** when [**loongsuite-util-genai**](https://pypi.org/project/loongsuite-util-genai/) and the community **opentelemetry-util-genai** are both pulled in or pinned differently.

**Step 3 — Run under `loongsuite-instrument`**

  Configure **where telemetry is exported** (see [Configure telemetry export](#configure-telemetry-export) below) using environment variables and/or `loongsuite-instrument` flags, then start your app:

  ```bash
  loongsuite-instrument \
    --traces_exporter console \
    --metrics_exporter console \
    --service_name demo \
    python demo.py
  ```

  For **programmatic** instrumentation, **install from source**, or **site-bootstrap** (`loongsuite-site-bootstrap`), see [Alternative installation methods](#alternative-installation-methods).

### Configure telemetry export

**Local debugging — console**

Use the SDK’s console exporters so traces/metrics/logs print to the terminal, for example via `loongsuite-instrument`:

```bash
loongsuite-instrument \
  --traces_exporter console \
  --metrics_exporter console \
  --logs_exporter console \
  python demo.py
```

Under the hood this aligns with **`ConsoleSpanExporter`**, **`ConsoleMetricExporter`**, and **`ConsoleLogRecordExporter`**.

**Remote / production — OTLP**

Before starting your application, install `opentelemetry-exporter-otlp`:

```bash
pip install opentelemetry-exporter-otlp
```

Point OpenTelemetry at a backend that accepts **OTLP** (gRPC or HTTP/protobuf), using **`OtlpSpanExporter`**, **`OtlpMetricExporter`**, **`OtlpLogExporter`** (or the equivalent env vars / `loongsuite-instrument` flags), for example:

```bash
export OTEL_SERVICE_NAME=demo
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317

loongsuite-instrument \
  --traces_exporter otlp \
  --metrics_exporter otlp \
  python demo.py
```

See also [OpenTelemetry environment variables](https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/) for `OTEL_EXPORTER_OTLP_*`.

## Alternative installation methods

<a id="alternative-installation-methods"></a>

If you are not using the [recommended `loongsuite-instrument` integration approach](#install-and-run-loongsuite), use **one** of the options below.

### Programmatic instrumentation

For applications where you can edit code and want explicit control over OpenTelemetry initialization.

**Step 1 — Install instrumentations** yourself from PyPI using the names in [Supported frameworks and components](#supported-frameworks-and-components).

  ```bash
  pip install loongsuite-instrumentation-agentscope
  ```
  
  > **Note:** If you need packages under [`instrumentation-genai/`](instrumentation-genai), use **Option A or B** together with **`loongsuite-distro`** / **`loongsuite-bootstrap`**. Relying only on manual `pip` can cause **dependency resolution conflicts** when [**loongsuite-util-genai**](https://pypi.org/project/loongsuite-util-genai/) and the community **opentelemetry-util-genai** are both pulled in or pinned differently.

**Step 2 — Initialize the OpenTelemetry SDK** before anything emits telemetry. You are wiring the same exporters as in [Configure telemetry export](#configure-telemetry-export).

  ```python
  from opentelemetry import metrics, trace
  from opentelemetry.sdk.metrics import MeterProvider
  from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
  from opentelemetry.sdk.resources import Resource
  from opentelemetry.sdk.trace import TracerProvider
  from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

  resource = Resource.create({"service.name": "demo"})
  tracer_provider = TracerProvider(resource=resource)
  tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
  trace.set_tracer_provider(tracer_provider)

  metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
  metrics.set_meter_provider(
      MeterProvider(resource=resource, metric_readers=[metric_reader])
  )
  ```

**Step 3 — Call the framework instrumentor**, then start your app without `loongsuite-instrument`.

  ```python
  from opentelemetry.instrumentation.agentscope import AgentScopeInstrumentor

  AgentScopeInstrumentor().instrument()
  # then import / start your agents (e.g. asyncio.run(main()))
  ```

### Install from source (for development)

**Step 1 — Clone this repository** and checkout your branch.

  ```bash
  git clone https://github.com/alibaba/loongsuite-python-agent.git
  ```

**Step 2 — Install upstream OpenTelemetry Python core and local LoongSuite components** from a Git checkout of [opentelemetry-python](https://github.com/open-telemetry/opentelemetry-python):

  ```bash
  cd loongsuite-python-agent
  GIT_ROOT="git+https://github.com/open-telemetry/opentelemetry-python.git"
  # Use ONE pip install command so resolver sees all constraints together;
  # split installs can downgrade/replace api+semconv when local editable deps are installed later.
  pip install \
    "${GIT_ROOT}#subdirectory=opentelemetry-api" \
    "${GIT_ROOT}#subdirectory=opentelemetry-semantic-conventions" \
    "${GIT_ROOT}#subdirectory=opentelemetry-sdk" \
    -e ./util/opentelemetry-util-genai \
    -e ./opentelemetry-instrumentation \
    -e ./loongsuite-distro
  ```

**Step 3 — Install the instrumentations you need**, for example:

  ```bash
  pip install -e ./instrumentation-loongsuite/loongsuite-instrumentation-agentscope
  ```

**Step 4 — Run under `loongsuite-instrument`**

  Configure **where telemetry is exported** (see [Configure telemetry export](#configure-telemetry-export) below) using environment variables and/or `loongsuite-instrument` flags, then start your app:

  ```bash
  loongsuite-instrument \
    --traces_exporter console \
    --metrics_exporter console \
    --service_name demo \
    python demo.py
  ```

### Site-bootstrap (Beta)

Run **without** changing codes or bootstrap commands: a **`.pth` hook** loads LoongSuite’s distro early (see [loongsuite-site-bootstrap/README.md](loongsuite-site-bootstrap/README.md)).

**Step 1 - Install LoongSuite Site Bootstrap** 

  ```bash
  pip install loongsuite-site-bootstrap
  ```

**Step 2 — Install instrumentations**

  ```bash
  loongsuite-bootstrap -a install --latest
  # for specific version: loongsuite-bootstrap -a install --version X.Y.Z
  ```

  If you want a different installation approach, see [Step 2 — Install instrumentations](#install-step-2-options) in [Install and run loongsuite](#install-and-run-loongsuite).

**Step 3 — Enable the hook**:

  ```bash
  export LOONGSUITE_PYTHON_SITE_BOOTSTRAP=True
  ```

**Step 4 — Create `~/.loongsuite/bootstrap-config.json`** with the OpenTelemetry environments keys you need.

  ```json
  {
    "OTEL_SERVICE_NAME": "demo",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4317",
    "OTEL_TRACES_EXPORTER": "otlp",
    "OTEL_METRICS_EXPORTER": "otlp"
  }
  ```

  Then run `python demo.py`. For **console** exporters, other backends, using **`loongsuite-instrument`** instead of plain `python`, or full precedence / edge cases, see [loongsuite-site-bootstrap/README.md](loongsuite-site-bootstrap/README.md).

> **Beta:** Site-bootstrap affects every Python process in the environment where it is enabled; read the package README before using it in production.

---

## Optional: OTLP examples

### AgentScope Studio

[AgentScope Studio](https://github.com/agentscope-ai/agentscope-studio) provides a web UI for traces and metrics.

```shell
pip install agentscope-studio
as_studio
```

Use the OTLP endpoint Studio prints (often `http://127.0.0.1:31415`), for example:

```shell
loongsuite-instrument \
    --traces_exporter otlp \
    --metrics_exporter otlp \
    --exporter_otlp_protocol http/protobuf \
    --exporter_otlp_endpoint http://127.0.0.1:31415 \
    --service_name demo \
    python demo.py
```

Or set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` / `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` accordingly. Details: [AgentScope Studio](https://github.com/agentscope-ai/agentscope-studio).

### Forward OTLP to Jaeger via LoongCollector

#### Launch Jaeger

```plaintext
docker run --rm --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 14250:14250 \
  -p 14268:14268 \
  -p 14269:14269 \
  -p 9411:9411 \
  jaegertracing/all-in-one:1.53.0
```

#### Launch LoongCollector

1. Install LoongCollector per its [documentation](https://observability.cn/project/loongcollector/quick-start/).
2. Add configuration under `conf/continuous_pipeline_config/local/oltp.yaml`:

```plaintext
enable: true
global:
  StructureType: v2
inputs:
  - Type: service_otlp
    Protocols:
      GRPC:
        Endpoint: 0.0.0.0:6666
flushers:
  - Type: flusher_otlp
    Traces:
      Endpoint: http://127.0.0.1:4317
```

3. Start LoongCollector, for example:

```plaintext
nohup ./loongcollector > stdout.log 2> stderr.log &
```

#### Run the demo against LoongCollector → Jaeger

```plaintext
loongsuite-instrument \
  --exporter_otlp_protocol grpc \
  --traces_exporter otlp \
  --exporter_otlp_insecure true \
  --exporter_otlp_endpoint 127.0.0.1:6666 \
  --service_name demo \
  python demo.py
```

Open the Jaeger UI and confirm traces arrive.

![Trace view in Jaeger](docs/_assets/img/quickstart-results.png)

## Community

We are looking forward to your feedback and suggestions. You can join
our [DingTalk group](https://qr.dingtalk.com/action/joingroup?code=v1,k1,mexukXI88tZ1uiuLYkKhdaETUx/K59ncyFFFG5Voe9s=&_dt_no_comment=1&origin=11?) or scan the QR code below to engage with us.

| LoongCollector SIG | LoongSuite Python SIG |
|----|----|
| <img src="docs/_assets/img/loongcollector-sig-dingtalk.jpg" height="150"> | <img src="docs/_assets/img/loongsuite-python-sig-dingtalk.jpg" height="150"> |

| LoongCollector Go SIG | LoongSuite Java SIG |
|----|----|
| <img src="docs/_assets/img/loongsuite-go-sig-dingtalk.png" height="150"> | <img src="docs/_assets/img/loongsuite-java-sig-dingtalk.jpg" height="150"> |

## Resources
* AgentScope: https://github.com/modelscope/agentscope
* Observability Community: https://observability.cn
