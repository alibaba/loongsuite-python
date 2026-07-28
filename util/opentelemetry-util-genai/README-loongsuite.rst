OpenTelemetry Util for GenAI - LoongSuite 扩展
=================================================

本文档描述 LoongSuite 对 OpenTelemetry GenAI Util 的扩展：适用范围、接入步骤与配置项。\ **对外发行**\ 时 PyPI 包名为 \ **loongsuite-otel-util-genai**\ ；Python 导入命名空间仍为 \ ``opentelemetry.util.genai``\ （与上游 GenAI Util 一致，见下节）。本仓库源码目录为 \ ``util/opentelemetry-util-genai``\ 。

.. note::

   旧发行名 ``loongsuite-util-genai`` 仍可供历史安装使用，但新的 LoongSuite GenAI
   工具库更新会发布到 ``loongsuite-otel-util-genai``。迁移时只需要更新 pip
   依赖名；Python import 仍保持 ``opentelemetry.util.genai``。

------------------------------------------------------------------------
1. 概述
------------------------------------------------------------------------

本仓库与 opentelemetry-util-genai 的关系
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

本模块在设计与演进上作为 OpenTelemetry 生态中 GenAI Util（包名 ``opentelemetry-util-genai``）的扩展：在兼容上游 API 与约定方向的前提下，由 LoongSuite **先行落地** 更丰富、更完整的 GenAI 语义与属性模型，并在合入社区前于本仓库迭代。覆盖范围包括 Agent、检索、记忆、入口与 ReAct 等场景。

**发行名称**：交付时，本模块以 **loongsuite-otel-util-genai** 发布至制品库（如 PyPI）；安装后提供的仍是 ``opentelemetry.util.genai`` 包及 ``ExtendedTelemetryHandler`` 等扩展接口，与从本 monorepo 构建安装的产物一致。

定位与能力
~~~~~~~~~~

面向大语言模型、Agent、工具调用与向量检索等场景，将调用语义映射为 OpenTelemetry 遥测数据：统一的 Span 命名与属性、可选的消息内容捕获、可选的 GenAI 事件，以及与 LoongSuite / OpenTelemetry 导出链路的对接。

本实现在上游 GenAI Util 能力之上提供扩展，主要包括：

- **llm**：聊天/补全类调用；支持多模态消息的**外置存储与 URI 替换**（见第 4 节），减轻 Trace 体积。
- **invoke_agent / create_agent**：Agent 调用与创建；``invoke_agent`` 可将
  Agent 名称写入 Baggage key ``gen_ai.agent.name``，使其下游 LLM、
  工具、检索、ReAct step 等 GenAI 子 Span 自动带上
  ``gen_ai.agent.name`` 属性。
- **embedding**：向量嵌入。
- **execute_tool**：工具/函数执行。
  - 当工具执行对应某个 skill 的加载动作时，可额外写入 ``gen_ai.skill.*`` 语义属性。
- **retrieval / rerank**：检索与重排序。
- **memory**：记忆读写等操作。
- **entry**：应用入口；可将 ``session_id`` / ``user_id`` 写入 Baggage，配合 ``BaggageSpanProcessor`` 做整条链路的染色。
- **react_step**：ReAct 单轮迭代标识。

相关操作对齐 **OpenTelemetry GenAI 语义约定** 并与上游 LLM 采集 API 衔接：``ExtendedTelemetryHandler`` 在基础 ``TelemetryHandler`` 之上扩展（详见上节与语义约定文档）。

------------------------------------------------------------------------
2. 快速使用
------------------------------------------------------------------------

请先安装发行包（见下节）。下文示例均使用 \ ``get_extended_telemetry_handler()``\ 获取单例 Handler；在首次获取前，全局 Tracer/Meter/Logger Provider 建议由 Instrumentation 自动注入，或由你在代码里显式完成 Provider 初始化（见 2.2 节）。

安装注意事项
~~~~~~~~~~~~

- **与上游包并存**：**loongsuite-otel-util-genai** 与社区发行 **opentelemetry-util-genai** 混装或重复指定时容易引发依赖解析冲突。**建议优先采用 LoongSuite 发行链路**，通过 ``loongsuite-instrument`` 及根目录 ``README.md`` 中的安装说明完成探针与本模块的组合安装与启动。
- **monorepo 本地安装探针时**：若 instrumentor 是从 **本仓库本地路径** 安装的（例如 ``pip install ./instrumentation-loongsuite/...``），则 **必须** 先从 **同一 monorepo 源码树** 安装本模块（例如 ``pip install -e ./util/opentelemetry-util-genai``）。否则安装 instrumentor 时，解析结果可能回落为**上游** GenAI Util，与本地探针版本不一致，导致扩展能力或行为不符合预期。

2.1. 使用 LoongSuite / OpenTelemetry Instrumentation 接入
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

前提：已通过 ``loongsuite-instrument``、``opentelemetry-instrument`` 或等价方式在进程启动时配置全局 Provider 与导出器，应用代码不自行构造 ``TracerProvider`` / ``MeterProvider``。

**安装**

::

    pip install loongsuite-otel-util-genai

从本 monorepo 本地安装时（与发行包等价源码树）::

    pip install -e ./util/opentelemetry-util-genai

Framework 探针若已声明对本包的依赖，会随探针一并安装；单独手写 GenAI Span 时仍需安装 \ **loongsuite-otel-util-genai**\ （或上述本地路径）。

**LLM 示例代码**

依赖 Instrumentation 把全局 Tracer/Meter/Logger Provider 配好；业务侧只需通过 OpenTelemetry API 确认 Provider 已就绪（可选），再拿 **扩展 Handler** 包一层 LLM 调用即可。

::

    from opentelemetry import trace
    from opentelemetry.util.genai.extended_handler import get_extended_telemetry_handler
    from opentelemetry.util.genai.types import InputMessage, LLMInvocation, OutputMessage, Text

    # 可选：确认当前进程已由 Instrumentation 设置 TracerProvider（未设置时多为 NoOp）
    _ = trace.get_tracer_provider()

    handler = get_extended_telemetry_handler()

    invocation = LLMInvocation(
        provider="openai",
        request_model="gpt-4o-mini",
        input_messages=[
            InputMessage(role="user", parts=[Text(content="用一句话解释 OpenTelemetry。")]),
        ],
    )
    with handler.llm(invocation) as inv:
        # 此处调用真实模型 API…
        inv.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content="OpenTelemetry 是用于可观测性的开放标准与实现。")],
                finish_reason="stop",
            )
        ]
        inv.input_tokens = 10
        inv.output_tokens = 20

未向 ``get_extended_telemetry_handler()`` 传入 ``tracer_provider`` / ``logger_provider`` 时，Handler 内部的 ``get_tracer`` / ``get_logger`` 会使用**当前全局** Provider（由 Instrumentation 注入）。

**环境变量**

若需在 Trace/Event 中采集**模型输入、输出等正文**（或完整消息结构），**必须**按 **2.3 节** 设置相关变量；否则默认不采集消息正文。

**进程启动（Instrumentation）**

使用本仓库推荐的启动方式时，请参阅 **仓库根目录** 的 ``README.md``（例如通过 ``loongsuite-instrument`` 指定导出器与 ``service_name`` 后再执行 ``python your_app.py``）。

若尚未安装 LoongSuite Instrumentation，请按根目录 ``README.md`` 的 **Quick start / INSTALL** 安装 ``opentelemetry-distro``、对应框架的 instrumentor，以及本 util 包，再使用包装命令启动进程。

2.2. 使用 OpenTelemetry SDK 直接接入
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

前提：不依赖进程级 auto-instrument，由应用入口显式注册 TracerProvider、MeterProvider、LoggerProvider 及导出器。

**安装**

::

    pip install loongsuite-otel-util-genai opentelemetry-sdk

按需增加导出器，例如 ``opentelemetry-exporter-otlp``。

**LLM 示例代码**

在获取 Handler **之前**初始化全局 Provider；创建 Handler 时把 **TracerProvider** 与 **LoggerProvider** 传入（便于 Span 与 GenAI Event；Metrics 使用 ``metrics.set_meter_provider`` 设置全局后，Handler 内 ``get_meter(..., meter_provider=None)`` 会使用全局 MeterProvider）。

::

    from opentelemetry import metrics, trace, _logs
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogRecordExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.util.genai.extended_handler import get_extended_telemetry_handler
    from opentelemetry.util.genai.types import InputMessage, LLMInvocation, OutputMessage, Text

    # 以下使用 Console*Exporter 便于本地调试。生产环境可将 SpanExporter / MetricExporter /
    # LogRecordExporter 替换为 OTLP、gRPC、Jaeger 等实现，以上报到 Collector 或兼容后端。
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())
    )

    metrics.set_meter_provider(
        MeterProvider(
            metric_readers=[PeriodicExportingMetricReader(ConsoleMetricExporter())]
        )
    )

    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(ConsoleLogRecordExporter())
    )
    _logs.set_logger_provider(logger_provider)

    handler = get_extended_telemetry_handler(
        tracer_provider=trace.get_tracer_provider(),
        logger_provider=logger_provider,
    )

    with handler.llm() as inv:
        inv.provider = "openai"
        inv.request_model = "gpt-4o-mini"
        inv.input_messages = [
            InputMessage(role="user", parts=[Text(content="你好")]),
        ]
        inv.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content="你好！")],
                finish_reason="stop",
            )
        ]

**环境变量**

若需在 Trace/Event 中采集**输入输出正文**，**必须**按 **2.3 节** 配置。

**进程启动（直接运行）**

::

    python your_app.py

无需 ``opentelemetry-instrument`` / ``loongsuite-instrument``，除非你还在同进程使用其他 auto-instrumentation。

2.3. 环境变量
~~~~~~~~~~~~~

本节是 GenAI 遥测是否**完整、可用**的关键：默认策略不将提示词、模型回复等消息正文写入遥测（``NO_CONTENT``），以降低敏感数据暴露与导出体积。若需要在可观测平台中排查或审计「输入与输出」正文，或以 Event 形式查看结构化消息，则必须以本节为准配置 ``OTEL_SEMCONV_STABILITY_OPT_IN``、``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``，并在选用 Event 路径时配合 ``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT`` 及 LoggerProvider（参见 2.2 节）。

``OTEL_SEMCONV_STABILITY_OPT_IN``

取值 ``gen_ai_latest_experimental`` 时启用 GenAI 实验性语义约定，便于与文档及导出器行为一致；未设置时部分属性可能不可用或与约定不一致。

``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``\ （消息内容落点）

- ``NO_CONTENT``：不采集消息正文（默认）。
- ``SPAN_ONLY``：正文写入 Span 属性（如 JSON 字符串）。
- ``EVENT_ONLY``：正文写入结构化 Event（需 LoggerProvider 及下方事件开关）。
- ``SPAN_AND_EVENT``：Span 与 Event 同时携带内容。

``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT``\ （GenAI Event）

- ``true``：在 ``EVENT_ONLY`` / ``SPAN_AND_EVENT`` 且已配置 LoggerProvider 时，按约定发出例如 ``gen_ai.client.*.operation.details`` 等事件。
- ``false``：不发出（默认）。

**配置示例**

::

    export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT
    export OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT=true

------------------------------------------------------------------------
3. Span 类型与用法
------------------------------------------------------------------------

约定：``handler = get_extended_telemetry_handler()``；在 ``with`` 块内仅修改 invocation 数据字段，勿直接修改 span。以下各节为典型用法（属性全集见语义约定与源码中的 attribute 写入逻辑）。

**llm** — ``handler.llm(invocation)`` 或 ``with handler.llm() as inv:``

::

    from opentelemetry.util.genai.types import LLMInvocation, InputMessage, OutputMessage, Text

    inv = LLMInvocation(provider="openai", request_model="gpt-4", input_messages=[InputMessage(role="user", parts=[Text(content="hi")])])
    with handler.llm(inv) as i:
        i.output_messages = [OutputMessage(role="assistant", parts=[Text(content="hello")], finish_reason="stop")]

**invoke_agent** — ``handler.invoke_agent()``

::

    from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text

    with handler.invoke_agent() as inv:
        inv.provider = "openai"
        inv.request_model = "gpt-4"
        inv.agent_name = "DemoAgent"
        inv.input_messages = [InputMessage(role="user", parts=[Text(content="任务说明")])]
        inv.output_messages = [OutputMessage(role="assistant", parts=[Text(content="回复")], finish_reason="stop")]

**create_agent** — ``handler.create_agent()``

::

    with handler.create_agent() as inv:
        inv.provider = "openai"
        inv.agent_name = "SupportAgent"
        inv.request_model = "gpt-4"

**embedding** — ``handler.embedding()``

::

    from opentelemetry.util.genai.extended_types import EmbeddingInvocation

    with handler.embedding() as inv:
        inv.provider = "openai"
        inv.request_model = "text-embedding-3-small"
        inv.dimension_count = 1536
        inv.input_tokens = 50

**execute_tool** — ``handler.execute_tool()``

::

    from opentelemetry.util.genai.extended_types import ExecuteToolInvocation

    with handler.execute_tool() as inv:
        inv.tool_name = "get_weather"
        inv.tool_call_arguments = {"city": "Beijing"}
        inv.tool_call_result = {"temp_c": 25}

当框架侧已经识别到该次工具调用对应某个 skill 的加载动作时，可在同一个
``execute_tool`` Span 上附加 skill 元信息：

::

    with handler.execute_tool() as inv:
        inv.tool_name = "read_file"
        inv.skill_name = "news"
        inv.skill_id = "workspace:default:news"
        inv.skill_description = "Read and summarize recent news."
        inv.skill_version = "1.0"

对应的 Span 属性为：

- ``gen_ai.skill.name``
- ``gen_ai.skill.id``
- ``gen_ai.skill.description``
- ``gen_ai.skill.version``

**retrieval** — \ ``handler.retrieval()``\ （Span kind 为 RETRIEVER）

::

    from opentelemetry.util.genai.extended_types import RetrievalDocument, RetrievalInvocation

    with handler.retrieval() as inv:
        inv.provider = "chroma"
        inv.data_source_id = "my_store"
        inv.query = "OpenTelemetry 是什么？"
        inv.top_k = 5
        inv.documents = [RetrievalDocument(id="d1", score=0.9, content="…", metadata={})]

**rerank** — ``handler.rerank()``

::

    from opentelemetry.util.genai.extended_types import RerankInvocation

    with handler.rerank() as inv:
        inv.provider = "cohere"
        inv.request_model = "rerank-v2"
        inv.rerank_input_documents = [{"id": "1", "text": "doc a"}, {"id": "2", "text": "doc b"}]
        inv.rerank_documents_count = 2
        inv.rerank_output_documents = [{"index": 0, "relevance_score": 0.95}]

**entry** — ``handler.entry(entry_inv)``；``session_id`` / ``user_id`` 须在构造 ``EntryInvocation`` 时传入，以便写入 Baggage。

::

    from opentelemetry.util.genai.extended_types import EntryInvocation

    entry_inv = EntryInvocation(session_id="sess-1", user_id="user-1")
    with handler.entry(entry_inv) as inv:
        inv.response_time_to_first_token = 1_000_000  # 纳秒

**react_step** — ``handler.react_step(step_inv)``

::

    from opentelemetry.util.genai.extended_types import ReactStepInvocation

    step_inv = ReactStepInvocation(round=1)
    with handler.react_step(step_inv) as step:
        with handler.llm() as llm_inv:
            llm_inv.provider = "openai"
            llm_inv.request_model = "gpt-4"
        step.finish_reason = "continue"

**memory** — ``handler.memory(MemoryInvocation(operation="add"))`` 等

::

    from opentelemetry.util.genai.extended_memory import MemoryInvocation

    mem = MemoryInvocation(operation="add")
    with handler.memory(mem) as inv:
        inv.user_id = "u1"
        inv.input_messages = "用户偏好：夜间模式"

------------------------------------------------------------------------
4. 多模态外置存储
------------------------------------------------------------------------

多模态 payload（图像、音频、视频等）体积较大，内联于 Span/Event 将显著增加存储与传输开销。本功能在导出遥测前将可处理部分**写入外部存储**，并在消息内把 ``Base64Blob``、``Blob`` 与部分 ``Uri`` 等多模态片段**替换**为目标存储 URI；Span 可附带 ``gen_ai.input.multimodal_metadata``、``gen_ai.output.multimodal_metadata`` 等扩展属性。

**组件与调用顺序**

- PreUploader（预处理器）：识别多模态 part，生成上传任务与目标 URI 布局（占位路径形如 ``BASE/DATE/HASH.ext``，对应实现中的 base_path、日期与文件扩展名布局），原地改写消息中的 part。
- **Uploader（上传器）**：异步消费上传任务，真正写入存储；失败只记日志，**不向业务抛异常**；相同内容可幂等跳过。
- 必须先 ``pre_upload`` 再 ``upload``；若任一 hook 未加载或返回 ``None``，整条多模态上传链路会降级为关闭（两者都为空）。

**环境变量**

- ``OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE``（默认 ``none``）

  - ``none``：关闭。
  - ``input`` / ``output`` / ``both``：分别处理请求侧、响应侧或两侧多模态。

- ``OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH``：存储根路径；当 mode 非 ``none`` 时**必须**配置。

**存储协议示例**：``file:///path/to/dir``、``memory://``、``oss://bucket/prefix``、``sls://project/logstore``，以及 fsspec 支持的其它协议。

**扩展环境变量**

- ``OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER`` / ``OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER``：hook 名称（默认 ``fs``）。
- ``OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED``：是否下载 http(s) URI 后再上传。
- ``OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_LOCAL_FILE_ENABLED`` + ``OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_ALLOWED_ROOT_PATHS``：是否允许读本地文件及允许路径前缀。
- ``OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_AUDIO_CONVERSION_ENABLED``：PCM 等转 WAV（需可选依赖 numpy/soundfile）。

**命令行示例**

::

    export OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE=both
    export OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH=file:///var/log/genai/multimodal

**依赖**

::

    pip install loongsuite-otel-util-genai[multimodal_upload]

音频转码还需安装可选依赖：先执行 ``pip install loongsuite-otel-util-genai``，再按 ``pyproject.toml`` 中 ``audio_conversion`` extra 的说明添加转码相关包（并打开对应环境变量）。

**资源释放**：启用多模态时，\ ``ExtendedTelemetryHandler``\ 在首次初始化时注册 \ ``atexit``\ ，进程退出时依次关闭 Handler / PreUploader / Uploader。常驻服务可显式调用 \ ``ExtendedTelemetryHandler.shutdown()``\ （见第 5 节）。

**自定义上传实现**：通过 ``pyproject.toml`` 中的 entry point ``opentelemetry_genai_multimodal_uploader``、``opentelemetry_genai_multimodal_pre_uploader`` 注册实现；本仓库默认提供 ``fs`` hook（见包内 ``pyproject.toml``）。

**运行时动态配置**

进程内维护一份多模态配置 snapshot（``MultimodalRuntimeConfig`` / ``MultimodalConfigSnapshot``）。启动时仍由上述环境变量完成 bootstrap；之后可通过公开 API 热更新，无需重启进程：

::

    from opentelemetry.util.genai._multimodal_upload import (
        get_multimodal_config_snapshot,
        update_multimodal_runtime_config,
    )

    update_multimodal_runtime_config(
        upload_mode="both",
        storage_base_path="file:///var/log/genai/multimodal",
        uploader_hook_name="fs",
        pre_uploader_hook_name="fs",
    )
    snapshot = get_multimodal_config_snapshot()

策略字段（如 ``upload_mode``、``download_enabled``、``local_file_enabled``、
``allowed_root_paths``）变更会递增 ``strategy_version``，立即作用于后续请求；
构造字段（如 ``storage_base_path``、uploader/pre-uploader hook 名）变更会递增
``uploader_generation``，并在下次上传前通过 ``get_or_rebuild_uploader_pair()``
热重建 uploader/pre-uploader。

**可选上传记录**：``FsUploader`` 终态成功/失败时会调用全局 ``MultimodalUsageRecorder``（默认 no-op）。需要时可通过 ``set_multimodal_usage_recorder()`` 注册自定义实现。

------------------------------------------------------------------------
5. 补充说明
------------------------------------------------------------------------

显式生命周期
~~~~~~~~~~~~~

推荐采用上下文管理器（``with handler.llm(...)`` 等）。需与既有控制流对接时，可使用 ``start_*`` / ``stop_*`` / ``fail_*``，例如 Agent 调用：

::

    from opentelemetry.util.genai.extended_types import InvokeAgentInvocation
    from opentelemetry.util.genai.types import Error

    inv = InvokeAgentInvocation(provider="openai")
    inv.request_model = "gpt-4"
    inv.agent_name = "A"
    handler.start_invoke_agent(inv)
    try:
        inv.input_messages = [...]
        inv.output_messages = [...]
        handler.stop_invoke_agent(inv)
    except Exception as e:
        handler.fail_invoke_agent(inv, Error(type=type(e), message=str(e)))

错误处理
~~~~~~~~

在 ``with`` 块内抛出异常时，Handler 会将 Span 标为错误并记录类型等信息；你也可以在业务层捕获后自行决定如何 **re-raise**。

::

    try:
        with handler.invoke_agent() as inv:
            inv.provider = "openai"
            inv.request_model = "gpt-4"
            raise RuntimeError("downstream failed")
    except RuntimeError:
        pass  # Span 上已反映错误；若需自定义 error 字段可改用 start/fail API

进程退出与上传收尾
~~~~~~~~~~~~~~~~~~

::

    from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

    ExtendedTelemetryHandler.shutdown()

可在 Web 框架或运行时的 shutdown 回调中调用，以便收尾异步上传等逻辑（以实现代码为准）。

------------------------------------------------------------------------
6. 参考
------------------------------------------------------------------------

- OpenTelemetry GenAI Utils 设计说明：`Design Document <https://docs.google.com/document/d/1w9TbtKjuRX_wymS8DRSwPA03_VhrGlyx65hHAdNik1E/edit?tab=t.qneb4vabc1wc#heading=h.kh4j6stirken>`_
- `OpenTelemetry 项目 <https://opentelemetry.io/>`_
- `OpenTelemetry GenAI 语义约定 <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
- LoongSuite Python Agent 仓库：`loongsuite-python <https://github.com/alibaba/loongsuite-python>`_（仓库根目录 ``README.md``：安装 Instrumentation 与 ``loongsuite-instrument`` 的 Quick start）
