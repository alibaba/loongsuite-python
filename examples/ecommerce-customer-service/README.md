# Text-only e-commerce customer service

This example shows a small **Chinese-language** LangGraph customer-service
workflow with LoongSuite automatic instrumentation. It accepts text only:
there is no image input, file upload, browser UI, or multimodal model.

It targets LangGraph 1.2+ and uses LangChain 1.x `create_agent`, the supported
LangGraph-backed agent API.

The workflow is:

```text
Chinese customer question
  -> intent_router
  -> presales_agent | aftersales_agent | clarify
  -> response_review
  -> final Chinese text response
```

The router, specialist prompts, synthetic data, tool results, CLI, and customer
responses use Simplified Chinese. The specialist Agent display names are
`售前服务` and `售后服务`, so their observable spans are named
`invoke_agent 售前服务` and `invoke_agent 售后服务`. Stable Python identifiers,
graph node names, tool names, and route values remain in English. The two
specialist branches use separate prompts and tools. All products, orders,
policies, and tool results are fictional fixtures in `tools.py`.

## Install

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Configure a model

The default uses the DashScope OpenAI-compatible endpoint and `qwen-plus`:

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

Override the compatible endpoint or model when needed:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
export MODEL_NAME="your-model"
```

`DASHSCOPE_API_KEY` takes precedence over `OPENAI_API_KEY`. Do not commit keys
to this directory.

## Run with LoongSuite

Console spans are convenient for a local smoke test:

```bash
export OTEL_SERVICE_NAME="ecommerce-customer-service"
export OTEL_TRACES_EXPORTER="console"
export OTEL_METRICS_EXPORTER="none"
export OTEL_LOGS_EXPORTER="none"
export OTEL_SEMCONV_STABILITY_OPT_IN="gen_ai_latest_experimental"

loongsuite-instrument python app.py \
  --question "云步通勤鞋适合日常步行吗？42 码有货吗？"
```

After-sales example:

```bash
loongsuite-instrument python app.py \
  --question "订单 DEMO-1001 昨天签收，鞋底有问题，应该怎么处理？"
```

Omit `--question` to start a simple interactive Chinese text loop.

For a detailed scenario walkthrough and a ready-to-use training script, see
[README.zh-CN.md](README.zh-CN.md#培训讲解指南).

With LangChain and LangGraph instrumentation enabled, a specialist request is
expected to include the router LLM, the selected Agent, ReAct Step, Tool/LLM
children, and the final reviewer LLM. Only the chosen specialist branch runs.
The outer graph nodes remain `presales_agent` and `aftersales_agent`, while the
inner Agent spans use the Chinese display names above.

## Test

The offline tests do not call an external model:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

They cover routing, separate specialist branches, synthetic tool results,
missing orders, fail-open behavior, and concurrent state isolation.

## Privacy boundary

This example is deliberately generic. `P-DEMO-*` products and `DEMO-*` orders
are synthetic and are not derived from a real store or customer environment.
