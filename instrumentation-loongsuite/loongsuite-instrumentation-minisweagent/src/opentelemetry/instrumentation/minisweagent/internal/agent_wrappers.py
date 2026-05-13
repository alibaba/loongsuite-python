"""wrapt hooks for DefaultAgent.run / DefaultAgent.step."""

from __future__ import annotations

from typing import Any, Callable

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import (
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    set_span_in_context,
)

from opentelemetry.instrumentation.minisweagent.config import (
    OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN,
)
from opentelemetry.instrumentation.minisweagent.internal.delegates import (
    GEN_AI_FRAMEWORK,
    GEN_AI_SPAN_KIND,
)


def _task_preview(task: str) -> str:
    if not task:
        return ""
    m = OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN
    if len(task) <= m:
        return task
    return task[: m - 3] + "..."


class DefaultAgentRunWrapper:
    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        instance._otel_msw_round = 0  # noqa: SLF001
        task = args[0] if args else kwargs.get("task", "") or ""
        agent_name = f"{instance.__class__.__module__}.{instance.__class__.__name__}"
        span_name = f"invoke_agent {agent_name}"
        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute(GEN_AI_SPAN_KIND, "AGENT")
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.INVOKE_AGENT.value,
            )
            span.set_attribute(GEN_AI_FRAMEWORK, "minisweagent")
            span.set_attribute(GenAI.GEN_AI_AGENT_NAME, agent_name)
            pv = _task_preview(str(task))
            if pv:
                span.set_attribute("minisweagent.task.preview", pv)
            try:
                result = wrapped(*args, **kwargs)
            except BaseException as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
            if isinstance(result, dict):
                es = result.get("exit_status")
                if es:
                    span.set_attribute("minisweagent.exit_status", str(es))
                sub = result.get("submission")
                if sub:
                    span.set_attribute(
                        "minisweagent.submission.preview",
                        _task_preview(str(sub)),
                    )
            return result


class DefaultAgentStepWrapper:
    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        from minisweagent.exceptions import InterruptAgentFlow  # noqa: PLC0415

        r = int(getattr(instance, "_otel_msw_round", 0) or 0) + 1
        instance._otel_msw_round = r  # noqa: SLF001
        span = self._tracer.start_span("react step", kind=SpanKind.INTERNAL)
        span.set_attribute(GEN_AI_SPAN_KIND, "STEP")
        span.set_attribute("gen_ai.operation.name", "react")
        span.set_attribute(GEN_AI_FRAMEWORK, "minisweagent")
        span.set_attribute("gen_ai.react.round", r)
        ctx = set_span_in_context(span)
        token = context_api.attach(ctx)
        try:
            return wrapped(*args, **kwargs)
        except InterruptAgentFlow as exc:
            span.set_attribute("gen_ai.react.finish_reason", type(exc).__qualname__)
            raise
        except BaseException as exc:
            span.set_attribute("gen_ai.react.finish_reason", type(exc).__qualname__)
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            context_api.detach(token)
            span.end()
