"""手动埋点（纯 SDK 方式）触发多模态「预授权 OSS 模式」上传的最小示例。

与自动埋点的区别：这里不依赖任何框架插桩，完全由业务代码自己构造
``LLMInvocation`` 并调用 ``ExtendedTelemetryHandler`` 的生命周期方法。

多模态上传能力挂在 handler 层（``ExtendedTelemetryHandler`` 继承了
``MultimodalProcessingMixin``），而不是挂在某个框架插桩里，所以纯手动埋点
同样会触发上传。

与「挂载 ARMS 探针」场景的区别：这里进程内没有探针，因此
- TracerProvider 需要本文件自己初始化；
- SLS project、presign endpoint、LicenseKey 都无处回落，必须由环境变量给出，
  否则整条多模态链路会静默降级为关闭（属性里仍是原始 base64/URL）。
  下面的 ``check_env()`` 就是为了把这种静默降级提前暴露成显式报错。

上传队列的优雅退出不依赖探针：``ExtendedTelemetryHandler`` 构造时就注册了
atexit drain，纯 SDK 方式同样生效。

运行方式见 ``../docs/manual-instrumentation.md``。
"""

import logging
import os
import struct
import sys
import zlib

# 多模态上传链路使用标准 logging。置 MULTIMODAL_DEBUG_LOG=1 可把上传过程打到
# stderr，便于分别确认 presign 申请与 OSS PUT 的结果。
if os.getenv("MULTIMODAL_DEBUG_LOG") == "1":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("opentelemetry.util.genai._multimodal_upload").setLevel(
        logging.DEBUG
    )

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.util.genai.extended_handler import (
    get_extended_telemetry_handler,
)
from opentelemetry.util.genai.types import (
    Blob,
    ContentCapturingMode,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)
from opentelemetry.util.genai.utils import (
    get_content_capturing_mode,
    is_experimental_mode,
)

ATTR_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "manual_trace_attr.log"
)

# 纯 SDK 场景下没有探针注入这些值，缺任何一项 presign 都会降级。
REQUIRED_ENV = (
    (
        "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE",
        "需为 both / span_only 之类的非 none 值，否则不会走上传",
    ),
    (
        "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER",
        "需为 presign",
    ),
    (
        "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_PROJECT",
        "对象地址里的 SLS project，无探针时无处回落",
    ),
    (
        "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_ENDPOINT",
        "申请预签名 URL 的 endpoint，无探针时无处回落",
    ),
)


def check_env() -> bool:
    """把「配置缺失导致的静默降级」提前变成显式报错。

    多模态上传只处理「已经进入 span 的消息内容」，所以内容捕获没打开时
    上传链路虽然装配正常，却没有任何 part 可传，表现为 span 里既没有
    messages 属性也没有任何上传日志——这一步就是为了先把它拦下来。
    """
    missing = [
        (name, why) for name, why in REQUIRED_ENV if not os.getenv(name)
    ]
    license_key = os.getenv(
        "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_LICENSE_KEY"
    ) or os.getenv("ARMS_LICENSE_KEY")
    if not license_key:
        missing.append(
            (
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_LICENSE_KEY",
                "presign 鉴权用，也可用 ARMS_LICENSE_KEY 提供",
            )
        )
    for name, why in missing:
        print(f"[demo] 缺少环境变量 {name}：{why}", file=sys.stderr)

    # 用库自身的判定，避免自己猜环境变量的合法取值。
    ok = not missing
    if not is_experimental_mode():
        print(
            "[demo] 需设置 OTEL_SEMCONV_STABILITY_OPT_IN="
            "gen_ai_latest_experimental，否则消息内容不会进 span",
            file=sys.stderr,
        )
        ok = False
    elif get_content_capturing_mode() not in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    ):
        # 该变量取的是 ContentCapturingMode 的枚举名，写 true/1 会被判为非法
        # 值并静默落到 NO_CONTENT。
        print(
            "[demo] 需设置 OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="
            "SPAN_ONLY 或 SPAN_AND_EVENT（取枚举名，不是 true）",
            file=sys.stderr,
        )
        ok = False
    return ok


def make_png(width: int = 96, height: int = 96, rgb=(220, 80, 40)) -> bytes:
    """生成一张合法的纯色 PNG，让示例不依赖真实模型或外部网络。"""
    raw = b""
    for _ in range(height):
        raw += b"\x00" + bytes(rgb) * width  # 每行前置 filter type 0

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class AttrDumpProcessor(SpanProcessor):
    """旁路 processor：把 span 终态属性写盘，用于确认 uri 是否被替换成 sls://。

    多模态 span 的 end 发生在异步 worker 线程里，这里拿到的是替换后的属性。
    仅用于示例自证，生产接入不需要。
    """

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        ctx = span.get_span_context()
        with open(ATTR_LOG, "a", encoding="utf-8") as handle:
            handle.write(
                f"[span] name={span.name} "
                f"trace_id={ctx.trace_id:032x} "
                f"span_id={ctx.span_id:016x}\n"
            )
            for key, value in (span.attributes or {}).items():
                handle.write(f"   attr {key} = {value}\n")
            handle.write("\n")


def build_tracer_provider() -> TracerProvider:
    """纯 SDK 方式自建 TracerProvider。

    真实接入时这里应挂 OTLPSpanExporter 把 span 发往后端；示例只挂旁路
    processor，因为要验证的是多模态属性替换，不是链路上报。
    """
    provider = TracerProvider(
        resource=Resource.create(
            {SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "manual-genai-demo")}
        )
    )
    provider.add_span_processor(AttrDumpProcessor())
    trace.set_tracer_provider(provider)
    return provider


def main() -> int:
    if not check_env():
        print("[demo] 配置不全，presign 会降级为关闭，已中止", file=sys.stderr)
        return 2

    provider = build_tracer_provider()

    # handler 是单例：首次获取时就绑定 TracerProvider。
    handler = get_extended_telemetry_handler(tracer_provider=provider)
    png = make_png()
    print(f"[demo] 待上传 PNG: {len(png)} bytes")

    # 1. 构造一次 LLM 调用，输入为纯文本
    invocation = LLMInvocation(
        request_model="manual-sdk-demo-model",
        provider="manual-demo",
        input_messages=[
            InputMessage(role="user", parts=[Text(content="画一张纯色方块图")])
        ],
    )

    # 2. 开始埋点：创建 span 并注入 context
    handler.start_llm(invocation)
    print("[demo] start_llm 完成")

    # 3. 模拟模型返回一张内联 PNG。
    #    Blob 走 inline bytes，无需开启 DOWNLOAD_ENABLED；
    #    若模型返回的是远端 URL，改用 Uri part 并开启该开关。
    invocation.output_messages = [
        OutputMessage(
            role="assistant",
            parts=[Blob(mime_type="image/png", modality="image", content=png)],
            finish_reason="stop",
        )
    ]
    invocation.response_id = "manual-demo-response-1"
    invocation.input_tokens = 12
    invocation.output_tokens = 0

    # 4. 结束埋点。含多模态时 stop_llm 不阻塞业务：
    #    上传与 span end 会被转入异步 worker。
    handler.stop_llm(invocation)
    print("[demo] stop_llm 完成（多模态已转异步处理）")

    # 5. 等异步上传收尾。
    #    handler 构造时已注册 atexit drain，进程正常退出也会等上传完成；
    #    这里显式 shutdown 是为了当场拿到确定结果，不依赖 atexit 的执行顺序。
    #    shutdown 之后该 handler 不应再复用。
    type(handler).shutdown()
    provider.force_flush(10_000)

    print(f"[demo] span 属性已写入 {ATTR_LOG}")
    print("[demo] 检查其中 gen_ai.output.messages 的 uri 是否为 sls:// 开头")
    return 0


if __name__ == "__main__":
    sys.exit(main())
