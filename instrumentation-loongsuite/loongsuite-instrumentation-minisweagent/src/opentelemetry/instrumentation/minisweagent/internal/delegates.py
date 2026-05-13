"""Tracing delegates for Environment (factory-injected wrappers).

LLM spans/metrics are intentionally NOT emitted here: the underlying
LiteLLM/OpenAI instrumentation already produces a high-quality LLM span
for each model call, so emitting another one from minisweagent would only
duplicate data.
"""

from __future__ import annotations

from typing import Any

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer, use_span

from opentelemetry.instrumentation.minisweagent.config import (
    OTEL_MINISWEAGENT_COMMAND_PREVIEW_MAX_LEN,
)

GEN_AI_SPAN_KIND = "gen_ai.span.kind"
GEN_AI_FRAMEWORK = "gen_ai.framework"


def _preview(text: str | None, max_len: int) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


class TracingEnvironment:
    """Delegates to an inner Environment and emits one TOOL span per ``execute`` call."""

    __slots__ = ("_inner", "_tracer")

    def __init__(self, inner: Any, tracer: Tracer):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_tracer", tracer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def execute(self, action: dict, cwd: str = "", **kwargs: Any) -> dict[str, Any]:
        from minisweagent.exceptions import InterruptAgentFlow  # noqa: PLC0415

        command = action.get("command", "") if isinstance(action, dict) else ""
        preview = _preview(str(command), OTEL_MINISWEAGENT_COMMAND_PREVIEW_MAX_LEN)
        inner = self._inner
        env_type = f"{inner.__class__.__module__}.{inner.__class__.__name__}"
        span_name = "execute_tool bash"
        span = self._tracer.start_span(span_name, kind=SpanKind.INTERNAL)
        span.set_attribute(GEN_AI_SPAN_KIND, "TOOL")
        span.set_attribute(
            GenAI.GEN_AI_OPERATION_NAME, GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value
        )
        span.set_attribute(GEN_AI_FRAMEWORK, "minisweagent")
        span.set_attribute(GenAI.GEN_AI_TOOL_NAME, "bash")
        span.set_attribute(GenAI.GEN_AI_TOOL_TYPE, "function")
        span.set_attribute("minisweagent.environment.class", env_type)
        if preview:
            span.set_attribute("minisweagent.command.preview", preview)

        with use_span(span, end_on_exit=False):
            try:
                result = self._inner.execute(action, cwd, **kwargs)
            except InterruptAgentFlow as exc:
                span.set_attribute("minisweagent.interrupt", type(exc).__qualname__)
                raise
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
            else:
                if isinstance(result, dict):
                    rc = result.get("returncode")
                    if rc is not None:
                        span.set_attribute("minisweagent.shell.returncode", int(rc))
                    exinf = result.get("exception_info")
                    if exinf:
                        span.set_attribute(
                            "minisweagent.shell.exception_info",
                            _preview(str(exinf), OTEL_MINISWEAGENT_COMMAND_PREVIEW_MAX_LEN),
                        )
                return result
            finally:
                span.end()
