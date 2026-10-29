OpenTelemetry Util for GenAI - LoongSuite 扩展
=================================================

GenAI Utils 是可独立安装和使用的开源 Python 工具库，为应用与框架提供
GenAI 调用的类型、生命周期和 OpenTelemetry 遥测辅助 API。
对外发行包名为 ``loongsuite-otel-util-genai``，Python 导入命名空间为
``opentelemetry.util.genai``；本仓库源码目录为 ``util/opentelemetry-util-genai``。
Provider 与导出器由使用方通过标准 OpenTelemetry SDK 配置；本库复用该配置生成遥测。
以下 API 与字段以当前源码版本为准。

.. contents:: 本页导航
   :local:
   :depth: 2

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

面向大语言模型、Agent、工具调用与向量检索等场景，将调用语义映射为 OpenTelemetry
遥测数据：统一的 Span 命名与属性、可选的消息内容捕获及 GenAI 事件。
使用方选择导出器，将数据发送到本地控制台、OpenTelemetry Collector 或兼容后端。

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

``ExtendedTelemetryHandler`` 在基础 ``TelemetryHandler`` 之上扩展。
LLM 等通用字段衔接 OpenTelemetry GenAI 语义约定；ENTRY、STEP、RERANKER
等为 LoongSuite 扩展，不能据此推断上游 util 或所有后端均支持相同能力。

------------------------------------------------------------------------
2. 快速使用
------------------------------------------------------------------------

安装工具库后，由应用初始化 OpenTelemetry SDK，再创建 Handler。
已配置 Provider 的应用可以直接复用现有配置。内容采集变量应在首次使用库前设置，
配置示例见 2.3 节。
下文统一采用 **显式构造 Invocation，再传给 Handler**：请求字段在开始前准备好，
业务调用在 ``with`` 内执行，响应字段在返回后补充。

2.1. 安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用标准 OpenTelemetry SDK 配置 Provider 时，安装：

::

    pip install loongsuite-otel-util-genai opentelemetry-sdk

从仓库根目录安装本地源码时：

.. code-block:: console

    pip install -e ./util/opentelemetry-util-genai opentelemetry-sdk

LoongSuite 发行包与社区发行包 ``opentelemetry-util-genai`` 提供相同的 Python
导入命名空间，应在同一个环境中选择一种实现，避免混装导致文件覆盖或版本冲突。

按需增加导出器，例如 ``opentelemetry-exporter-otlp``。

2.2. 初始化 OpenTelemetry 并创建 Handler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在应用初始化阶段配置 Provider 与导出器，然后获取 Handler。
Trace 需要 TracerProvider 与 SpanExporter；Metrics、GenAI Event 分别需要
MeterProvider、LoggerProvider 及对应导出器。已有全局 Provider 时直接复用，
也可以在创建 Handler 时显式传入 Provider。

**可直接运行的本地示例**

先按 2.3 节启用 ``SPAN_ONLY``，将下例保存为 ``your_app.py`` 后运行
``python your_app.py``。示例只对合成文档做本地关键词排序，无需模型密钥或后端；
它演示工具库的手动埋点与本地导出。

::

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    from opentelemetry.util.genai.extended_handler import (
        ExtendedTelemetryHandler,
        get_extended_telemetry_handler,
    )
    from opentelemetry.util.genai.extended_types import EntryInvocation, RerankInvocation

    tracer_provider = TracerProvider(
        resource=Resource.create({"service.name": "genai-manual-demo"}),
    )
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tracer_provider)
    handler = get_extended_telemetry_handler(tracer_provider=tracer_provider)

    def call_reranker(query, documents):
        results = [
            {"index": i, "relevance_score": float(query.lower() in doc.lower())}
            for i, doc in enumerate(documents)
        ]
        return {"results": sorted(results, key=lambda item: item["relevance_score"], reverse=True)}

    query = "OpenTelemetry"
    documents = ["OpenTelemetry collects traces.", "A synthetic unrelated document."]
    try:
        entry = EntryInvocation(session_id="demo-session", user_id="demo-user")
        with handler.entry(entry):
            invocation = RerankInvocation(
                provider="local-demo",
                request_model="keyword-demo",
                input_documents=[
                    {"id": str(i), "text": doc} for i, doc in enumerate(documents)
                ],
                documents_count=len(documents),
            )
            with handler.rerank(invocation) as inv:
                response = call_reranker(query, documents)
                inv.output_documents = response["results"]
                inv.attributes["biz.rerank_strategy"] = "keyword"
    finally:
        ExtendedTelemetryHandler.shutdown()
        tracer_provider.shutdown()

控制台应有两个相同 trace ID 的 Span：``rerank_documents keyword-demo`` 的
``parent_id`` 指向 ``enter_ai_application_system``。Rerank 包含
``gen_ai.rerank.documents.count=2``、``gen_ai.provider.name=local-demo``、
``gen_ai.request.model=keyword-demo`` 和 ``biz.rerank_strategy=keyword``；
``SPAN_ONLY`` 下还包含两个 documents 属性。改为 ``NO_CONTENT`` 并在新进程重跑时，
这两个 documents 属性应消失，计数和上述元数据仍保留。
Entry 的身份字段要复制为子 Span 属性，还需 5.1 节的 BaggageSpanProcessor。

``get_extended_telemetry_handler()`` 缓存单例，首次创建后的调用不会重新绑定
Provider，即使再次传入不同的 Provider；应在应用初始化阶段完成配置。
未传入 ``tracer_provider`` / ``logger_provider`` 时使用全局 Provider。
仅调用 ``trace.get_tracer_provider()`` 不能证明已配置 SDK 和导出器。

**按需增加 Metrics / Event**

以下代码也需放在首次获取 Handler **之前**，并在应用退出时关闭新增的 Provider。
``get_extended_telemetry_handler`` 仅接受 ``tracer_provider``、``logger_provider``；
MeterProvider 通过全局 API 设置。需要独立 Provider 组合时，可以自行构造
``ExtendedTelemetryHandler(tracer_provider=..., meter_provider=..., logger_provider=...)``。

.. code-block:: python

    from opentelemetry import metrics, _logs
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.util.genai.extended_handler import get_extended_telemetry_handler

    # 兼容新旧 OpenTelemetry SDK 的控制台日志导出器名称。
    try:
        from opentelemetry.sdk._logs.export import ConsoleLogRecordExporter
    except ImportError:
        from opentelemetry.sdk._logs.export import ConsoleLogExporter as ConsoleLogRecordExporter

    meter_provider = MeterProvider(
        metric_readers=[PeriodicExportingMetricReader(ConsoleMetricExporter())],
    )
    metrics.set_meter_provider(meter_provider)
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(ConsoleLogRecordExporter()))
    _logs.set_logger_provider(logger_provider)

    handler = get_extended_telemetry_handler(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
    )

当前直接 Histogram 记录器只处理 LLM；其他扩展操作的直接 Metrics
记录器仍为占位实现。后端从 Span 派生的指标是另一条链路。
配置 MeterProvider 不表示所有 Span 类型都会产生直接指标。
Console 导出用于本地检查；生产环境按所需信号配置 OTLP 等导出器。

2.3. 环境变量
~~~~~~~~~~~~~

默认 ``NO_CONTENT`` 不导出受内容开关控制的消息正文。模型、提供方、文档计数、
已观测到的 usage 等元数据不等于正文，不需要通过开启正文采集来保留。
配置应在 SDK、Handler 初始化前完成；实验性语义开关在首次使用时缓存，
修改后请重启进程。

``OTEL_SEMCONV_STABILITY_OPT_IN``

包含 ``gen_ai_latest_experimental`` 时启用本文的实验性内容与事件路径，
可与其他 opt-in 值逗号分隔。仅设置内容模式而不启用此项，不会启用正文采集。

``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``\ （消息内容落点）

.. list-table:: 内容模式（已启用实验性语义）
   :header-rows: 1
   :widths: 22 26 30 22

   * - 模式
     - Span 正文
     - Event 正文（有事件实现的操作）
     - 未显式配置时的事件开关
   * - ``NO_CONTENT`` （默认）
     - 无
     - 无
     - ``false``
   * - ``SPAN_ONLY``
     - 有
     - 无
     - ``false``
   * - ``EVENT_ONLY``
     - 无
     - 有
     - ``true``
   * - ``SPAN_AND_EVENT``
     - 有
     - 有
     - ``true``

``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT``\ （GenAI Event）

- 显式 ``true`` / ``false`` 优先于表中默认值。``false`` 关闭事件，
  因而 ``EVENT_ONLY`` 加 ``false`` 不会把正文转存到 Span。
- ``true`` 在 ``NO_CONTENT`` / ``SPAN_ONLY`` 下也可以发出元数据事件，
  但不会绕过 Event 正文的内容开关。
- Event 通过 OpenTelemetry Logs API 导出，需要 LoggerProvider、LogRecordProcessor
  与日志导出器；它不是 ``span.add_event``，Trace 导出器不能代替日志导出器。
- 当前事件实现覆盖 ``llm``（``gen_ai.client.inference.operation.details``）、
  ``invoke_agent``（``gen_ai.client.agent.invoke.operation.details``）和
  ``memory``（``gen_ai.memory.operation.details``）。其他类型没有对应的内容事件；
  Rerank 文档、工具参数/结果、Entry 消息、检索正文需用 ``SPAN_ONLY`` 或
  ``SPAN_AND_EVENT`` 查看，``EVENT_ONLY`` 不会为它们补发正文事件。
- Retrieval 在 ``NO_CONTENT`` / ``EVENT_ONLY`` 下仍可记录文档 ``id`` / ``score``，
  不包含 ``content`` / ``metadata`` 或查询正文。

**本地排查推荐配置**

::

    export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
    export OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT=false

需要事件时再按表切换模式并配置日志导出器。``SPAN_AND_EVENT`` 会同时导出两份内容，
应评估体积与存储成本。内容开关不负责全局脱敏：``attributes``、错误消息与其他
不受该开关控制的字段（例如当前 Rerank 的 ``scoring_prompt``）需由调用方控制，
不要写入密钥、原始客户数据或完整 SDK 对象。自定义属性边界见 3.1 节。

------------------------------------------------------------------------
3. Span 类型与用法
------------------------------------------------------------------------

3.1. 通用写法与自定义属性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下片段假设已经执行 ``handler = get_extended_telemetry_handler()``。
``call_*``、``search_documents`` 等是待替换的业务函数，不是 util 提供的 API；
各例说明了所用返回结构。只有 2.2 节是无需补充业务实现即可运行的完整程序。

- **调用前**：构造本次调用独立的 Invocation，准备 provider、模型、输入等已知字段。
  每次调用创建新对象，不复用已经结束的 Invocation，也不在并发请求间共享它。
- **调用中**：在 ``with handler.<operation>(invocation) as inv`` 内执行业务。
  **调用后、退出前**：补充实际输出、已观测到的 usage 和自定义元数据。
  不直接操作 Handler 管理的 ``span``、``context_token``。
- **字段名校验**：构造器遇到未知关键字会报 ``TypeError``，能尽早发现拼写错误；
  类型注解并不进行完整的运行时校验。``inv.some_new_field = ...`` 可能成为普通
  Python 动态属性，但 Handler 不会自动导出它。
- **自定义属性入口**：在构造时用 ``attributes={...}``，或在生命周期内修改
  ``inv.attributes``。它们作为 Span 属性写出，不会自动传播为 Baggage、复制到
  子 Span，或成为指标维度；事件也不保证包含它们。

.. code-block:: python

    from opentelemetry.util.genai.extended_types import RerankInvocation

    invocation = RerankInvocation(
        provider="example-reranker",
        attributes={"biz.rerank_strategy": "semantic"},
    )
    with handler.rerank(invocation) as inv:
        response = call_reranker(query, documents)
        inv.output_documents = response["results"]
        inv.attributes["biz.result_count"] = len(response["results"])

自定义属性使用自己的命名空间（如 ``biz.*``），不要覆盖 ``gen_ai.*`` 等内置语义。
值应为 OpenTelemetry 属性支持的字符串、布尔、整数、浮点或同类型的一维序列，
不要直接传字典、嵌套对象或 ``None``。``attributes`` 没有正文开关或自动脱敏保护：
将文档、提示词、工具结果写到这里会绕过 ``NO_CONTENT``，也可能造成无界载荷。
正文使用下列内置字段；自定义属性只放经过筛选、有界的业务元数据。
LLM 的额外指标维度有单独的 ``metric_attributes``，只能使用低基数值。

3.2. 操作与字段速查
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

表中 ``gen_ai.span.kind`` 是 LoongSuite 语义属性，不是 OpenTelemetry
``SpanKind`` 枚举，例如 retrieval 的 OTel SpanKind 为 ``INTERNAL``。
字段全集见 `基础类型 <https://github.com/alibaba/loongsuite-python/blob/main/util/opentelemetry-util-genai/src/opentelemetry/util/genai/types.py>`_、
`扩展类型 <https://github.com/alibaba/loongsuite-python/blob/main/util/opentelemetry-util-genai/src/opentelemetry/util/genai/extended_types.py>`_ 和
`Memory 类型 <https://github.com/alibaba/loongsuite-python/blob/main/util/opentelemetry-util-genai/src/opentelemetry/util/genai/extended_memory/memory_types.py>`_。

.. list-table:: 显式 Invocation 与主要字段
   :header-rows: 1
   :widths: 18 28 16 38

   * - Handler 方法
     - Invocation 类型
     - ``gen_ai.span.kind``
     - 主要字段
   * - ``llm``
     - ``types.LLMInvocation``
     - ``LLM``
     - ``provider``, ``request_model``, ``input_messages``, ``output_messages``, usage
   * - ``invoke_agent``
     - ``extended_types.InvokeAgentInvocation``
     - ``AGENT``
     - ``provider``, ``agent_name``, ``input_messages``, ``output_messages``
   * - ``create_agent``
     - ``extended_types.CreateAgentInvocation``
     - ``AGENT``
     - ``provider``, ``agent_name``, ``agent_id``
   * - ``embedding``
     - ``extended_types.EmbeddingInvocation``
     - ``EMBEDDING``
     - ``request_model``, ``provider``, ``dimension_count``, ``input_tokens``
   * - ``execute_tool``
     - ``extended_types.ExecuteToolInvocation``
     - ``TOOL``
     - ``tool_name``, ``tool_call_id``, ``tool_call_arguments``, ``tool_call_result``
   * - ``retrieval``
     - ``extended_types.RetrievalInvocation``
     - ``RETRIEVER``
     - ``data_source_id``, ``query``, ``top_k``, ``documents``
   * - ``rerank``
     - ``extended_types.RerankInvocation``
     - ``RERANKER``
     - ``provider``, ``request_model``, ``input_documents``, ``documents_count``, ``output_documents``
   * - ``entry``
     - ``extended_types.EntryInvocation``
     - ``ENTRY``
     - ``session_id``, ``user_id``, ``input_messages``, ``output_messages``
   * - ``react_step``
     - ``extended_types.ReactStepInvocation``
     - ``STEP``
     - ``round``（从 1 开始）, ``finish_reason``
   * - ``memory``
     - ``extended_memory.MemoryInvocation``
     - ``MEMORY``
     - ``operation``, ``user_id``, ``input_messages``, ``output_messages``

3.3. LLM 与 Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**llm**：完成 2.2 节的 Provider 初始化后，通过 Handler 记录模型调用。
下例中的 ``call_model`` 是业务函数的占位名：接收提示词，返回包含 ``text``、
``finish_reason`` 的字典，可选包含 SDK 实际返回的 ``input_tokens`` / ``output_tokens``。
消息使用 ``InputMessage``、``OutputMessage`` 与 ``Text`` 等类型，
按实际 SDK 响应映射字段，不要把 SDK 原始响应整体传入消息字段。
``input_tokens`` / ``output_tokens`` 只从实际 usage 映射；缓存 Token 是细分项，
不要再叠加到已经包含它们的输入 Token 中；未知 usage 保持 ``None``。

.. code-block:: python

    from opentelemetry.util.genai.extended_handler import get_extended_telemetry_handler
    from opentelemetry.util.genai.types import InputMessage, LLMInvocation, OutputMessage, Text

    handler = get_extended_telemetry_handler()
    prompt = "用一句话解释 OpenTelemetry。"
    invocation = LLMInvocation(
        provider="example-provider",  # 替换为实际提供方与模型
        request_model="example-model",
        input_messages=[InputMessage(role="user", parts=[Text(content=prompt)])],
    )
    with handler.llm(invocation) as inv:
        response = call_model(prompt)
        inv.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content=response["text"])],
                finish_reason=response["finish_reason"],
            ),
        ]
        inv.input_tokens = response.get("input_tokens")
        inv.output_tokens = response.get("output_tokens")

**invoke_agent**：``call_agent`` 返回 ``{"text": ..., "finish_reason": ...}``。
Agent 输入描述本次请求，输出只记录最终答案；中间工具结果属于工具调用。
``agent_name`` 在开始时还用于 Baggage 传播，应在构造时给出。

.. code-block:: python

    from opentelemetry.util.genai.extended_types import InvokeAgentInvocation
    from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text

    task = "解释 OpenTelemetry 的用途。"
    invocation = InvokeAgentInvocation(
        provider="example-provider",
        agent_name="DemoAgent",
        input_messages=[InputMessage(role="user", parts=[Text(content=task)])],
    )
    with handler.invoke_agent(invocation) as inv:
        response = call_agent(task)
        inv.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content=response["text"])],
                finish_reason=response["finish_reason"],
            ),
        ]

**create_agent**：仅在实际创建 Agent 时记录，不能替代 Agent 执行。
``create_remote_agent`` 返回包含 ``id`` 的字典。

.. code-block:: python

    from opentelemetry.util.genai.extended_types import CreateAgentInvocation

    invocation = CreateAgentInvocation(
        provider="example-provider", agent_name="SupportAgent",
    )
    with handler.create_agent(invocation) as inv:
        response = create_remote_agent("SupportAgent")
        inv.agent_id = response["id"]

3.4. Embedding 与工具
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**embedding**：``call_embedding`` 返回 ``{"vector": [...], "input_tokens": ...}``；
Token 数可缺省。维度由实际向量长度获得，不向遥测写入整个向量。

.. code-block:: python

    from opentelemetry.util.genai.extended_types import EmbeddingInvocation

    invocation = EmbeddingInvocation(
        provider="example-provider", request_model="example-embedding",
    )
    with handler.embedding(invocation) as inv:
        response = call_embedding("OpenTelemetry")
        inv.dimension_count = len(response["vector"])
        inv.input_tokens = response.get("input_tokens")

**execute_tool**：``get_weather`` 返回可 JSON 序列化的结果。
若存在真实工具调用 ID，将其写入 ``tool_call_id``，与模型返回的工具调用对应。

.. code-block:: python

    from opentelemetry.util.genai.extended_types import ExecuteToolInvocation

    invocation = ExecuteToolInvocation(
        tool_name="get_weather",
        tool_call_arguments={"city": "Beijing"},
    )
    with handler.execute_tool(invocation) as inv:
        inv.tool_call_result = get_weather(city="Beijing")

当工具确实加载某个 skill 时，可以在构造 ``ExecuteToolInvocation`` 时附带
``skill_name``、``skill_id``、``skill_description``、``skill_version``，分别写入
``gen_ai.skill.name``、``gen_ai.skill.id``、``gen_ai.skill.description``、
``gen_ai.skill.version``；普通文件读取无需虚构 skill 元数据。

3.5. Retrieval 与 Rerank
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**retrieval**：``search_documents(query, top_k=...)`` 返回业务文档字典列表，
本例约定每项包含 ``id``、``score``、``text``；显式映射为 ``RetrievalDocument``。

.. code-block:: python

    from opentelemetry.util.genai.extended_types import RetrievalDocument, RetrievalInvocation

    query = "OpenTelemetry 是什么？"
    invocation = RetrievalInvocation(
        provider="example-store", data_source_id="my_store", query=query, top_k=5,
    )
    with handler.retrieval(invocation) as inv:
        results = search_documents(query, top_k=5)
        inv.documents = [
            RetrievalDocument(id=item["id"], score=item["score"], content=item["text"])
            for item in results
        ]

**rerank**：``call_reranker(query, documents)`` 返回 ``{"results": [...]}``，
每项结果包含原文档索引与实际相关性分数。正文应先按业务策略筛选和限制大小，
再放入内置 documents 字段；未知分数不补零。

.. code-block:: python

    from opentelemetry.util.genai.extended_types import RerankInvocation

    invocation = RerankInvocation(
        provider="example-reranker",
        request_model="example-rerank-model",
        input_documents=[
            {"id": str(i), "text": doc[:200]} for i, doc in enumerate(documents)
        ],
        documents_count=len(documents),
    )
    with handler.rerank(invocation) as inv:
        response = call_reranker(query, documents)
        inv.output_documents = response["results"]
        inv.attributes["biz.rerank_strategy"] = "semantic"

.. list-table:: Python 字段与导出属性的对应关系
   :header-rows: 1
   :widths: 26 40 34

   * - Python 字段
     - Span 属性
     - 采集条件
   * - ``input_documents``
     - ``gen_ai.rerank.input_documents``
     - 实验性语义 + Span 正文模式
   * - ``documents_count``
     - ``gen_ai.rerank.documents.count``
     - 值不为 ``None``，不依赖正文开关
   * - ``output_documents``
     - ``gen_ai.rerank.output_documents``
     - 实验性语义 + Span 正文模式

Python 字段 **没有** ``rerank_`` 前缀。把 Span 属性名的后半段直接当 Python
字段名，例如 ``inv.rerank_input_documents``，不会被 Handler 识别。

3.6. Entry、ReAct Step 与 Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**entry**：``handle_request`` 返回业务最终答案字符串。身份字段在构造时传入，
才能在开始时放入 Baggage。嵌套调用与身份传播配置见 5.1 节。

.. code-block:: python

    from opentelemetry.util.genai.extended_types import EntryInvocation
    from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text

    prompt = "解释 OpenTelemetry 的用途。"
    invocation = EntryInvocation(
        session_id="demo-session",
        user_id="demo-user",
        input_messages=[InputMessage(role="user", parts=[Text(content=prompt)])],
    )
    with handler.entry(invocation) as inv:
        answer = handle_request(prompt)
        inv.output_messages = [
            OutputMessage(
                role="assistant", parts=[Text(content=answer)], finish_reason="stop",
            ),
        ]

``response_time_to_first_token`` 只在实际观测到首个输出时填写，单位为纳秒；
不要将秒或毫秒数直接赋入该字段。

**react_step**：只描述框架实际执行的一轮 Reasoning/Acting。
``run_one_step`` 执行本轮业务并返回包含 ``finish_reason`` 的字典；
应在实际下游调用处记录 LLM、工具等子操作，不为展示树形而创建空 LLM Span。

.. code-block:: python

    from opentelemetry.util.genai.extended_types import ReactStepInvocation

    invocation = ReactStepInvocation(round=1)
    with handler.react_step(invocation) as inv:
        result = run_one_step()
        inv.finish_reason = result["finish_reason"]

**memory**：``save_memory`` 写入记忆并返回可序列化的结果。
``MemoryInvocation`` 来自 ``extended_memory``，不在 ``extended_types`` 中。

.. code-block:: python

    from opentelemetry.util.genai.extended_memory import MemoryInvocation

    content = "演示偏好：夜间模式"
    invocation = MemoryInvocation(
        operation="add", user_id="demo-user", input_messages=content,
    )
    with handler.memory(invocation) as inv:
        inv.output_messages = save_memory(user_id="demo-user", content=content)

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

- ``OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH``：存储根路径；
  使用默认 ``fs`` 后端且 mode 非 ``none`` 时必须配置。

**存储协议示例**：``file:///path/to/dir``、``memory://``，以及对应 fsspec
文件系统实现支持的协议。使用其他存储系统时，由应用安装相应实现并配置访问凭据。

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

    pip install 'loongsuite-otel-util-genai[multimodal_upload]'

音频转码还需 ``pip install 'loongsuite-otel-util-genai[audio_conversion]'``，
并打开对应环境变量。多模态正文处理还要求启用实验性语义及 Span 正文模式；
仅设置上传模式不会在 ``NO_CONTENT`` / ``EVENT_ONLY`` 下触发该处理链路。

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

可选上传后端
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

上传后端通过 entry point 加载，其配置独立于 Trace、Metrics 和 Logs 的 Provider。
应用根据所选存储系统配置地址、认证和可选依赖；默认 ``fs`` 后端可用于本地文件
或相应 fsspec 文件系统支持的存储。

包中还提供 ``presign`` 后端，用于连接已部署的兼容预签名上传服务。
该后端的协议、认证和配置项见
`presign 后端实现 <https://github.com/alibaba/loongsuite-python/blob/main/util/opentelemetry-util-genai/src/opentelemetry/util/genai/_multimodal_upload/presign_uploader.py>`_。
其他存储服务也可以通过相同的 uploader / pre-uploader 接口扩展。

------------------------------------------------------------------------
5. 上下文与生命周期
------------------------------------------------------------------------

5.1. 父子关系与身份传播
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在父操作的 ``with`` 内开始子操作，当前 OTel Context 决定父子关系。
``EntryInvocation.session_id`` / ``user_id`` 在开始时进入 Baggage；要使它们成为
子 Span 的属性，还需配置 BaggageSpanProcessor。身份染色与父子关系是两回事。
``InvokeAgentInvocation.agent_name`` 在开始时进入 Baggage，后续 GenAI 子操作
会从中读取 Agent 名称；在 ``with`` 内才赋值无法补做之前的传播。

::

    pip install opentelemetry-processor-baggage

在创建子 Span 前，把处理器添加到应用已有的 TracerProvider：

.. code-block:: python

    from opentelemetry.processor.baggage import BaggageSpanProcessor

    tracer_provider.add_span_processor(
        BaggageSpanProcessor(
            lambda key: key in {"gen_ai.session.id", "gen_ai.user.id"},
        ),
    )

在应用初始化阶段注册一次处理器。处理器按白名单复制 Baggage；
不要把凭据或敏感正文放入 Baggage，它可能被传播到下游服务。

5.2. 异步、并发与流式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Handler 提供的是同步上下文管理器。协程中可以在同一个 task 内
  ``with handler.llm(invocation)``，并在块内 ``await`` 业务调用；不使用
  ``async with``。每个并发 task 自己创建和结束 Invocation。
- 子任务应在父作用域内创建并等待完成，再退出父作用域。不要在父 Span 结束后
  才消费返回的协程或流，也不要让后台任务继续沿用已经结束的请求生命周期。
  线程池或跨进程调用应使用相应的 OTel Context 传播机制，不能仅靠传递 Invocation。
- ``start_*`` 可接收 ``context=...`` 指定父 Context；``with`` 方法没有该参数。
  开始和结束须在同一个 task/Context 中配对，不能把附带 attach token 的
  Invocation 交给另一个 task 去 ``stop_*``。
- 包住“创建流对象”的调用只覆盖创建时间。流 Span 应覆盖实际消费直到结束，
  也要处理提前退出、``close`` / ``aclose``、异常和取消。不要为遥测额外拉取 chunk。
  生成器在 ``yield`` 时必须恢复调用方 Context；通用流代理应由经过生命周期验证的
  流式适配器实现，不能直接在跨 ``yield`` 的 ``with`` 中长期激活 Context。

这些约束用于设计和排查，不能仅通过改成显式 Invocation 就证明时序问题已经修复。
样本若出现子 Span 在父 Span 结束后才开始，应继续核对 task 创建/等待、流消费位置、
父 Context 和 Span 时间戳；父子 ID 相连本身不能证明生命周期正确。

5.3. 显式生命周期与错误处理
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

优先使用本文的 ``with`` 写法。对接框架原生回调时，可采用
``start_<operation>(invocation, context=None)``，成功后
``stop_<operation>(invocation)``，失败时
``fail_<operation>(invocation, Error(type=type(exc), message=...))``。
``Error`` 从 ``opentelemetry.util.genai.types`` 导入，``type`` 接收异常类而非字符串。
一次调用只选择一种收尾路径，不在 ``with`` 内再次手动 stop/fail，也不重复结束。

正常工作的上下文管理器会记录普通 ``Exception`` 并重新抛出。不要照搬吞掉业务
异常的 ``except: pass``；错误消息也可能含敏感内容，应由接入层控制。
当前管理器仅捕获 ``Exception``，不能据此承诺所有取消等 ``BaseException``
路径都已自动清理。自建异步/回调适配器需要单独验证取消、部分初始化失败和上下文恢复。

Handler 管理遥测生命周期。自建集成时，将异常隔离限制在遥测解析、记录与清理
步骤，业务调用与流迭代保留原有的错误处理语义。收尾失败不得覆盖原业务异常，
更不能重试业务调用；需要验证遥测失败时的清理和后续调用。

5.4. 进程退出与上传收尾
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

应用 shutdown 回调先结束业务调用，收尾多模态队列，再关闭自己持有的 SDK
Provider。不要在每个请求或每个 chunk 后调用 shutdown/force_flush。

.. code-block:: python

    from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

    ExtendedTelemetryHandler.shutdown()
    tracer_provider.shutdown()

若还创建了 MeterProvider、LoggerProvider，也应由应用在退出时调用它们的
``shutdown()``。Handler 的 shutdown 不会代替 SDK Provider 的导出收尾。

------------------------------------------------------------------------
6. 接入排查与版本边界
------------------------------------------------------------------------

**先验证本地导出**：运行 2.2 节，在 ``SPAN_ONLY`` / ``NO_CONTENT`` 两个新进程中
对照属性，再按需接入 OpenTelemetry Collector 或兼容后端。控制台可见只证明本地导出；后端是否收到，
需要按本次服务名、trace ID 和时间窗口读回验证。

.. list-table:: 常见现象
   :header-rows: 1
   :widths: 28 72

   * - 现象
     - 优先核对
   * - 完全没有 Span
     - SDK Provider、SpanProcessor/导出器、采样，以及 Handler 首次获取时机。
   * - Rerank 有 Span 但没有计数
     - 是否使用 ``documents_count`` 并传入非 ``None`` 值；计数不受正文开关控制。
   * - 有计数但没有 Rerank documents
     - 实验性语义开关、Span 正文模式，以及 ``input_documents`` / ``output_documents`` 字段。
   * - ``EVENT_ONLY`` 下没有工具或检索正文
     - 这些操作没有对应内容事件；按 2.3 节选择支持的内容落点。
   * - 事件不可见
     - 该操作是否实现事件、事件开关、LoggerProvider 与日志导出器；不能只查 Trace 导出器。
   * - 自定义字段不可见
     - 是否写入 ``inv.attributes``、值类型是否有效、是否在结束前赋值；动态 Python 属性不会自动导出。
   * - 子 Span 没有用户/会话属性
     - Entry 身份字段是否在开始前设置、BaggageSpanProcessor 是否注册且白名单包含相应键。
   * - 有父子 ID，但时间范围异常
     - 按 5.2 节核对异步/流式生命周期，不能用修改字段名推断修复。
   * - Token 或 Span 重复
     - 同一次 LLM 调用是否被多次记录、是否重复收尾、是否重复合计缓存 Token。

**记录实际运行版本与导入来源**：在运行应用的同一个 Python 环境中执行：

.. code-block:: python

    from importlib.metadata import PackageNotFoundError, version
    from opentelemetry.util.genai import extended_handler, extended_types

    for name in (
        "loongsuite-otel-util-genai", "loongsuite-util-genai",
        "opentelemetry-util-genai", "opentelemetry-api", "opentelemetry-sdk",
    ):
        try:
            print(name, version(name))
        except PackageNotFoundError:
            continue
    print("handler:", extended_handler.__file__)
    print("types:", extended_types.__file__)

这段输出供本地排查；对外提交时隐藏用户名等机器路径。
包版本存在不证明实际加载了它提供的文件，要结合导入来源判断。

------------------------------------------------------------------------
7. 参考
------------------------------------------------------------------------

- 当前实现：`ExtendedTelemetryHandler <https://github.com/alibaba/loongsuite-python/blob/main/util/opentelemetry-util-genai/src/opentelemetry/util/genai/extended_handler.py>`_、
  `扩展属性映射 <https://github.com/alibaba/loongsuite-python/blob/main/util/opentelemetry-util-genai/src/opentelemetry/util/genai/extended_span_utils.py>`_、
  `内容与事件配置 <https://github.com/alibaba/loongsuite-python/blob/main/util/opentelemetry-util-genai/src/opentelemetry/util/genai/utils.py>`_。
- `OpenTelemetry Python 手动埋点 <https://opentelemetry.io/docs/languages/python/instrumentation/>`_
- OpenTelemetry GenAI Utils 设计说明：`Design Document <https://docs.google.com/document/d/1w9TbtKjuRX_wymS8DRSwPA03_VhrGlyx65hHAdNik1E/edit?tab=t.qneb4vabc1wc#heading=h.kh4j6stirken>`_
- `OpenTelemetry 项目 <https://opentelemetry.io/>`_
- `OpenTelemetry GenAI 语义约定 <https://github.com/open-telemetry/semantic-conventions-genai>`_
- `LoongSuite GenAI 语义规范 <https://github.com/alibaba/loongsuite-semantic-conventions-genai>`_
- `GenAI Utils 源码与包信息 <https://github.com/alibaba/loongsuite-python/tree/main/util/opentelemetry-util-genai>`_
