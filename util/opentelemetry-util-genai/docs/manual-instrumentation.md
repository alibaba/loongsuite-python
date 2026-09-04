# GenAI 手动埋点（纯 SDK）使用文档

适用于**不挂载任何探针、也不依赖框架自动插桩**的场景：业务代码直接调用
`opentelemetry-util-genai` 的 handler 完成 GenAI 语义埋点。

自动埋点与手动埋点共用同一个 handler，因此两者能力一致 —— 包括多模态外置
存储。区别只是「谁来构造 invocation 对象」。

- 自动埋点：插桩包拦截框架调用后构造
- 手动埋点：业务代码自己构造

---

## 1. 安装

```bash
pip install opentelemetry-util-genai opentelemetry-sdk
# 多模态「预授权 OSS 模式」还需要 httpx
pip install httpx
```

## 2. 最小可用示例

手动埋点需要自己初始化 TracerProvider（没有探针帮你做这件事）。

### 2.1 初始化

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from opentelemetry.util.genai.extended_handler import get_extended_telemetry_handler
from opentelemetry.util.genai.types import (
    InputMessage, LLMInvocation, OutputMessage, Text,
)

provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "my-app"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

# handler 是单例：首次获取时就绑定 TracerProvider，之后传参不再生效。
handler = get_extended_telemetry_handler(tracer_provider=provider)

invocation = LLMInvocation(
    request_model="qwen-plus",
    provider="dashscope",
    input_messages=[InputMessage(role="user", parts=[Text(content="你好")])],
)

```

### 2.2 上下文管理器（推荐）

`with` 块正常退出时结束 span，抛异常时自动标记错误，不会漏掉 `stop`：

```python
with handler.llm(invocation) as inv:
    # ... 调用模型 ...
    inv.output_messages = [
        OutputMessage(role="assistant",
                      parts=[Text(content="你好，有什么可以帮你？")],
                      finish_reason="stop")
    ]
    inv.input_tokens = 12
    inv.output_tokens = 8
```

其余操作同理：`handler.embedding(...)`、`handler.execute_tool(...)`、
`handler.invoke_agent(...)` 等。

### 2.3 显式生命周期（与既有控制流对接）

已有的 try/except 结构不方便改成 `with` 时，用三元组手动配对：

```python
handler.start_llm(invocation)            # 创建 span、注入 context
try:
    # ... 调用模型 ...
    invocation.output_messages = [
        OutputMessage(role="assistant",
                      parts=[Text(content="你好，有什么可以帮你？")],
                      finish_reason="stop")
    ]
    invocation.input_tokens = 12
    invocation.output_tokens = 8
    handler.stop_llm(invocation)         # 写属性并结束 span
except Exception as exc:
    from opentelemetry.util.genai.types import Error
    handler.fail_llm(invocation, Error(message=str(exc), type=type(exc)))
    raise
```

要点：

- `start_*` / `stop_*` / `fail_*` 必须配对，否则 span 不会结束。
- 结果通过**修改 invocation 对象的字段**回填，而不是作为 `stop_*` 的参数。
- 两种写法完全等价，多模态上传都会正常触发（`with` 内部就是调用这三个方法）。

## 3. 让消息内容进入 span（两个必需开关）

默认**不采集**消息内容。多模态上传只处理「已进入 span 的消息内容」，所以这
两个开关没配对时，多模态链路装配正常却没有任何 part 可传 —— 表现为 span 里
既没有 `gen_ai.input.messages`，也没有任何上传日志。

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```

- 第一个必须包含 `gen_ai_latest_experimental`，否则内容采集整体不生效。
- 第二个取的是枚举名，合法值为 `NO_CONTENT` / `SPAN_ONLY` / `EVENT_ONLY` /
  `SPAN_AND_EVENT`。写 `true`、`1` 这类值会被判为非法并**静默**落回
  `NO_CONTENT`（只有一条 warning 日志）。

## 4. 支持的 span 类型

`ExtendedTelemetryHandler` 对以下操作提供 `start_* / stop_* / fail_*` 三元组，
用法与 `*_llm` 一致，只是 invocation 类型不同：

| 操作 | 方法前缀 | invocation 类型 |
| --- | --- | --- |
| LLM 调用 | `*_llm` | `LLMInvocation` |
| 向量化 | `*_embedding` | `EmbeddingInvocation` |
| 工具调用 | `*_execute_tool` | `ExecuteToolInvocation` |
| Agent 创建 / 调用 | `*_create_agent` / `*_invoke_agent` | `CreateAgentInvocation` / `InvokeAgentInvocation` |
| 检索 / 重排 | `*_retrieval` / `*_rerank` | `RetrievalInvocation` / `RerankInvocation` |
| 记忆 | `*_memory` | `MemoryInvocation` |
| 入口 | `*_entry` | `EntryInvocation` |
| ReAct 步骤 | `*_react_step` | `ReactStepInvocation` |

这些类型分散在三个模块里，注意导入路径：

- `opentelemetry.util.genai.types`：`LLMInvocation`、`Error`，以及消息与 part
  类型 `InputMessage` / `OutputMessage` / `Text` / `Blob` / `Uri`
- `opentelemetry.util.genai.extended_types`：上表其余 invocation
- `opentelemetry.util.genai.extended_memory.memory_types`：`MemoryInvocation`

---

## 5. 多模态外置存储：预授权 OSS 模式（presign）

图片、音频这类大对象不适合直接塞进 span 属性。开启后 handler 会把 part 内容
上传到对象存储，并把属性里的内容替换成 `sls://` 地址。

「预授权」指探针/SDK **不持有任何 OSS AK/SK**：用 LicenseKey 向服务端申请一个
短时效的预签名 URL，再直接 PUT 到服务端持有的 bucket。

### 5.1 配置

纯 SDK 场景没有探针注入身份，以下每一项都必须显式配置，缺任意一项会导致整条
多模态链路**降级为关闭**（属性里保留原始 base64/URL，不会产生不可访问的地址）。

```bash
# 启用多模态 + 选择 presign 通道
export OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE=both
export OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER=presign
export OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER=presign

# 身份：无探针时必须自己给
export OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_LICENSE_KEY='<your-license-key>'
export OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_WORKSPACE='<your-cms-workspace>'

# 申请预签名 URL 的 endpoint：无探针时无处回落，必须显式配置
export OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_ENDPOINT=<region>.log.aliyuncs.com

# 对象归属：project 必填；logstore 必须是该 project 下已存在的 logstore
export APSARA_APM_COLLECTOR_MULTIMODAL_SLS_PROJECT=proj-xtrace-xxx-cn-hangzhou
export APSARA_APM_COLLECTOR_MULTIMODAL_SLS_LOGSTORE=multimodal

# 可选：给本应用所有对象加统一前缀
export OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_PATH_PREFIX=genai/my-app
```

身份两项支持回落：`PRESIGN_LICENSE_KEY` / `PRESIGN_WORKSPACE` 未设置时，分别
回落到 `ARMS_LICENSE_KEY` / `ARMS_WORKSPACE`。这样同机挂载了 ARMS 探针时无需
重复配置，独立使用时也能自己指定。

`OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_BUCKET` 在本模式下**已废弃且被忽略**
（bucket 由服务端决定）。

### 5.2 对象地址

地址格式与 `arms` 模式同构，且**在发起上传前就能本地确定**：

```
sls://{project}/{logstore}[/{prefix}]/{yyyymmdd}/{md5}.{ext}
```

实际落在服务端 bucket 里的 object key 是它去掉 `sls://` 后的部分，前面再拼上
服务端选定的 bucket。属性里写入的地址与真实 object key 逐段一致。

`logstore` 未配置时缺省为 `logstore-multimodal`，该缺省值同时作用于地址与
presign 请求体，两者不会分叉。

### 5.3 异步与优雅退出

含多模态的调用里，`stop_llm` **不阻塞业务**：上传与 span end 会转入异步 worker。

`ExtendedTelemetryHandler` 构造时已注册 atexit drain，进程正常退出会等待上传
收尾，纯 SDK 方式同样生效。若想在确定的时间点收口（例如短生命周期脚本），可
显式调用：

```python
type(handler).shutdown()   # drain worker、pre_uploader、uploader
provider.force_flush(10_000)
```

`shutdown()` 之后该 handler 不应再复用。

### 5.4 可运行示例

`examples/multimodal_presign_manual.py` 是一个不依赖真实模型和外部网络的完整
示例（自己生成一张 PNG），并内置配置自检，把「配置缺失导致的静默降级」提前
变成显式报错。

```bash
cd util/opentelemetry-util-genai
export PYTHONPATH="$PWD/src"
# ... 按 5.1 配好环境变量，另加 ...
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
export MULTIMODAL_DEBUG_LOG=1     # 打印 presign 申请与 OSS PUT 的结果

python examples/multimodal_presign_manual.py
```

正常输出会包含两次 HTTP 200（presign 申请 + OSS PUT），并在
`examples/manual_trace_attr.log` 里看到 `gen_ai.output.messages` 的 `uri` 已被
替换成 `sls://` 地址。

### 5.5 如何确认真的传成功了

**属性被替换不等于上传成功** —— 地址是本地生成的，上传失败时也可能已经写入。
判定要看这两条：

1. 日志里 presign 申请和 OSS PUT 都是 2xx（`MULTIMODAL_DEBUG_LOG=1`）。
2. span 属性里的 `sls://` 地址，去掉协议头后与 PUT 请求的 object key 逐段一致。

## 6. 故障排查

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| span 里没有 `gen_ai.*.messages`，也没有上传日志 | 内容采集未开启 | 按第 3 节配齐两个开关，注意第二个取枚举名 |
| 有 warning 说 `CAPTURE_MESSAGE_CONTENT` 值非法 | 写成了 `true`/`1` | 改为 `SPAN_ONLY` 或 `SPAN_AND_EVENT` |
| `Presign request rejected (status=404)`，响应体 `LogStoreNotExist` | logstore 在该 project 下不存在（常见于沿用了缺省值 `logstore-multimodal`） | 显式配置 `APSARA_APM_COLLECTOR_MULTIMODAL_SLS_LOGSTORE` 为真实存在的 logstore |
| `status=401/403` | LicenseKey 或 workspace 与 project 不属于同一租户 | 核对三者是否同一实例 |
| 上传链路整体没启动 | project、endpoint 或 LicenseKey 缺失导致降级 | 开 DEBUG 日志看降级原因；或参考示例的 `check_env()` 做启动自检 |
| span 属性仍是 base64 | 上传未在 span end 前完成 | 检查是否在 `shutdown()` / atexit drain 之前就强杀了进程 |

## 7. 已知限制

- `PRESIGN_ENDPOINT` 在纯 SDK 场景没有第二来源，必须显式配置；挂载 ARMS 探针
  时才能留空复用其 endpoint 探活结果。
- 身份与 endpoint 类配置只从环境变量读取，**不支持**运行时动态配置改写。
- `Blob`（inline bytes）无需额外开关；模型返回远端 URL 时要用 `Uri` part 并
  开启 `OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED`。
