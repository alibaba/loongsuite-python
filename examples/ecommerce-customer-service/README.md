# Text-only e-commerce customer service

This example shows a small LangGraph customer-service workflow with LoongSuite
automatic instrumentation. It accepts text only: there is no image input, file
upload, browser UI, or multimodal model.

The workflow is:

```text
question
  -> intent_router
  -> presales_agent | aftersales_agent | clarify
  -> response_review
  -> final text response
```

The two specialist branches use separate prompts and tools. All products,
orders, policies, and tool results are fictional fixtures in `tools.py`.

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
  --question "Are the CloudStep commuter shoes suitable for walking, and is size 42 available?"
```

After-sales example:

```bash
loongsuite-instrument python app.py \
  --question "Order DEMO-1001 was delivered yesterday and has a sole defect. What should I do?"
```

Omit `--question` to start a simple interactive text loop.

With LangChain and LangGraph instrumentation enabled, a specialist request is
expected to include the router LLM, the selected Agent, ReAct Step, Tool/LLM
children, and the final reviewer LLM. Only the chosen specialist branch runs.

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
