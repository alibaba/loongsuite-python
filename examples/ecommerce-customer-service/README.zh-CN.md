# 纯文字电商客服

本示例展示一个使用 LoongSuite 自动埋点的小型 LangGraph 电商客服工作流。
示例只接受文字：不包含图片输入、文件上传、浏览器 UI 或多模态模型。

工作流如下：

```text
用户问题
  -> intent_router
  -> presales_agent | aftersales_agent | clarify
  -> response_review
  -> 最终文字回复
```

售前和售后分支具有独立的提示词和工具。`tools.py` 中的商品、订单、政策及
工具结果全部是虚构数据。

## 安装

需要使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 配置模型

默认使用 DashScope OpenAI-compatible 接口和 `qwen-plus`：

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

也可以替换为其他兼容接口：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
export MODEL_NAME="your-model"
```

`DASHSCOPE_API_KEY` 的优先级高于 `OPENAI_API_KEY`。请勿把密钥提交到仓库。

## 使用 LoongSuite 运行

本地验证可先把 Span 输出到控制台：

```bash
export OTEL_SERVICE_NAME="ecommerce-customer-service"
export OTEL_TRACES_EXPORTER="console"
export OTEL_METRICS_EXPORTER="none"
export OTEL_LOGS_EXPORTER="none"
export OTEL_SEMCONV_STABILITY_OPT_IN="gen_ai_latest_experimental"

loongsuite-instrument python app.py \
  --question "CloudStep 通勤鞋适合步行吗？42 码有货吗？"
```

售后示例：

```bash
loongsuite-instrument python app.py \
  --question "订单 DEMO-1001 昨天签收，鞋底有问题，应该怎么处理？"
```

省略 `--question` 后会进入简单的交互式文字问答。

启用 LangChain 和 LangGraph 埋点后，一次专业客服请求应包含路由 LLM、
被选中的 Agent、ReAct Step、Tool/LLM 子节点以及最终审查 LLM；未选中的
专业 Agent 不会执行。

## 测试

离线测试不会请求外部模型：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

测试覆盖路由、两个独立专业分支、虚构工具结果、订单未命中、fail-open
行为和并发状态隔离。

## 隐私边界

本示例刻意保持通用。`P-DEMO-*` 商品和 `DEMO-*` 订单均为虚构数据，
不来源于任何真实店铺或客户环境。
