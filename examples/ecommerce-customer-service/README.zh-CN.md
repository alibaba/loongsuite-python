# 纯文字电商客服

本示例展示一个使用 LoongSuite 自动埋点的小型中文 LangGraph 电商客服工作流。
示例只接受文字：不包含图片输入、文件上传、浏览器 UI 或多模态模型。

示例面向 LangGraph 1.2+，并使用 LangChain 1.x 的 `create_agent`，即当前受支持的
LangGraph Agent API。

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

运行时的意图识别提示词、专业 Agent 提示词、工具说明、虚构数据、异常兜底及
最终回答均使用简体中文。为了保持代码规范和 Trace 名称稳定，Python 标识符、
节点名和内部路由值仍使用英文。

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
  --question "云步通勤鞋适合日常步行吗？42 码有货吗？"
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

## 培训讲解指南

### 场景故事

把这个 Demo 理解成一家虚构店铺的“客服总台”。总台先判断用户是在买东西前
咨询，还是已经下单后的问题，再把请求转给有独立职责和工具的专业客服。
专业客服查到事实并生成草稿后，最后还有一名审核员统一语气并检查是否编造。

### 五步工作流

| 步骤 | 节点 | 作用 | 典型可观测信号 |
| --- | --- | --- | --- |
| 1 | `intent_router` | LLM 结构化识别售前、售后或意图不明 | 一次路由 LLM 调用 |
| 2 | 条件路由 | 只选择一个专业分支；低置信度进入澄清 | LangGraph 节点和边 |
| 3 | `presales_agent` / `aftersales_agent` | 独立 ReAct Agent 思考并选择工具 | Agent、ReAct Step、LLM |
| 4 | 虚构工具 | 查询商品/库存，或订单/政策/问题处理建议 | Tool Span 及工具参数、结果 |
| 5 | `response_review` | LLM 根据草稿和工具证据润色终审 | 最终审查 LLM 调用 |

售前 Agent 可以使用 `search_product_catalog`、`query_product_knowledge` 和
`check_inventory`；售后 Agent 可以使用 `lookup_order_history`、
`query_after_sales_policy` 和 `assess_issue`。两个工具集完全隔离，能够直观展示
“不同 Agent 有不同职责和能力边界”。

### 建议的现场演示

1. 售前问题：`云步通勤鞋适合日常步行吗？42 码有货吗？`
   讲解路由为什么选择售前，以及 Agent 如何组合商品知识和库存工具。
2. 售后问题：`订单 DEMO-1001 昨天签收，鞋底有问题，应该怎么处理？`
   讲解 Agent 为什么必须先查订单，再查询政策或给出人工核验步骤。
3. 模糊问题：`能帮帮我吗？`
   展示低置信度请求不会误进专业 Agent，而是先向用户澄清。

在可观测控制台中，可以沿一次请求查看“入口/路由 LLM → 被选中的 Agent →
ReAct Step → Tool 与专业 LLM → 最终审查 LLM”。培训时重点说明：未选中的
Agent 不执行；工具证据被传给最终审核员；任何模型或专业 Agent 异常都会返回
有限的中文兜底回答，而不是影响应用进程。

### 为什么这样设计

- **职责清晰**：售前和售后分别维护提示词与工具，不让一个 Agent 拥有所有权限。
- **过程可解释**：意图、路由、工具证据和最终回复都能在 Trace 中逐层观察。
- **事实有边界**：最终审核员只允许保留工具证据支持的信息，降低幻觉风险。
- **适合演示**：纯文字、虚构数据、单进程即可运行，不依赖真实电商系统。

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
