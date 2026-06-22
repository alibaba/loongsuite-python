# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrap DeerFlow Sandbox APIs as TOOL / TASK spans.

* ``Sandbox.execute_command`` / ``read_file`` / ``write_file`` / ``glob`` /
  ``grep`` / ``list_dir`` (abstract base methods) are wrapped as TOOL spans
  via ``ExecuteToolInvocation``. Patching the ABC automatically covers every
  subclass (``LocalSandbox`` and remote providers).
* ``SandboxProvider.acquire`` / ``acquire_async`` / ``release`` are wrapped as
  lightweight TASK spans (lifecycle events, no payload).
"""

from __future__ import annotations

import logging
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.deer_flow.utils import (
    DEER_FLOW_COMPONENT,
    DEER_FLOW_OPERATION,
    DEER_FLOW_TASK_NAME,
    _safe_call,
    _should_capture_content,
)
from opentelemetry.trace import SpanKind, Status, StatusCode, get_tracer
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.extended_types import ExecuteToolInvocation
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)

_SANDBOX_MODULE = "deerflow.sandbox.sandbox"
_SANDBOX_CLASS = "Sandbox"
_PROVIDER_MODULE = "deerflow.sandbox.sandbox_provider"
_PROVIDER_CLASS = "SandboxProvider"

# Map abstract method names to human-friendly tool names for the TOOL span.
_TOOL_NAME_MAP = {
    "execute_command": "bash",
    "read_file": "read_file",
    "write_file": "write_file",
    "glob": "glob",
    "grep": "grep",
    "list_dir": "list_dir",
}

_SANDBOX_METHODS = list(_TOOL_NAME_MAP.keys())


class _SandboxMethodWrapper:
    """Wrap a ``Sandbox`` abstract method as a TOOL span.

    ``ExtendedTelemetryHandler`` does not expose ``start_task``/``stop_task``;
    the Sandbox tool operations fit the TOOL span kind (``execute_tool``)
    cleanly, so we use ``ExecuteToolInvocation`` here per execute.md §3.4.4.
    """

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        method_name = getattr(wrapped, "__name__", "")
        tool_name = _TOOL_NAME_MAP.get(method_name, method_name or "sandbox_tool")
        sandbox_id = _safe_call(
            "get_sandbox_id", lambda: getattr(instance, "_id", None)
        )
        tool_call_id = f"sandbox:{sandbox_id}" if sandbox_id else f"sandbox:{tool_name}"

        capture = _should_capture_content()
        if capture:
            arguments: Any = self._extract_arguments(method_name, args, kwargs)
        else:
            arguments = "<redacted>"

        invocation = ExecuteToolInvocation(
            provider="deer-flow",
            tool_name=tool_name,
            tool_type="function",
            tool_call_id=tool_call_id,
            tool_call_arguments=arguments,
            attributes={
                DEER_FLOW_OPERATION: f"sandbox.{method_name}",
                DEER_FLOW_COMPONENT: "sandbox",
            },
        )

        started = _safe_call(
            "start_execute_tool",
            self._handler.start_execute_tool,
            invocation,
        )
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            if started:
                _safe_call(
                    "fail_execute_tool",
                    self._handler.fail_execute_tool,
                    invocation,
                    Error(message=str(exc) or type(exc).__name__, type=type(exc)),
                )
                span = getattr(invocation, "span", None)
                if span is not None and span.is_recording():
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

        if started:
            invocation.tool_call_result = (
                result if capture else "<redacted>"
            )
            _safe_call(
                "stop_execute_tool",
                self._handler.stop_execute_tool,
                invocation,
            )
        return result

    @staticmethod
    def _extract_arguments(
        method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if method_name == "execute_command":
            command = args[0] if args else kwargs.get("command")
            return {"command": command} if command is not None else None
        if method_name == "read_file":
            path = args[0] if args else kwargs.get("path")
            return {"path": path} if path is not None else None
        if method_name == "write_file":
            path = args[0] if args else kwargs.get("path")
            content = args[1] if len(args) > 1 else kwargs.get("content")
            append = args[2] if len(args) > 2 else kwargs.get("append", False)
            return {"path": path, "content": content, "append": append}
        if method_name == "glob":
            path = args[0] if args else kwargs.get("path")
            pattern = args[1] if len(args) > 1 else kwargs.get("pattern")
            return {"path": path, "pattern": pattern}
        if method_name == "grep":
            return {"args": list(args), "kwargs": dict(kwargs)} if kwargs else list(args)
        if method_name == "list_dir":
            path = args[0] if args else kwargs.get("path")
            return {"path": path} if path is not None else None
        return None


class _ProviderLifecycleWrapper:
    """Wrap ``SandboxProvider.acquire`` / ``release`` as TASK spans.

    Created manually because ``ExtendedTelemetryHandler`` has no
    ``start_task``/``stop_task``. TASK span kind + ``run_task`` operation name
    follows the LoongSuite semantic convention for Task spans.
    """

    def __init__(self, tracer: Any):
        self._tracer = tracer

    def _run(self, wrapped: Any, instance: Any, args: Any, kwargs: Any, *, method_name: str) -> Any:
        task_name = f"sandbox.{method_name}"
        span_name = f"run_task {task_name}"
        span = self._tracer.start_span(name=span_name, kind=SpanKind.INTERNAL)
        span.set_attribute(GEN_AI_SPAN_KIND, "TASK")
        span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "run_task")
        span.set_attribute(DEER_FLOW_OPERATION, f"sandbox.{method_name}")
        span.set_attribute(DEER_FLOW_COMPONENT, "sandbox_provider")
        span.set_attribute(DEER_FLOW_TASK_NAME, task_name)
        thread_id = kwargs.get("thread_id") if kwargs else None
        if thread_id is None and args:
            thread_id = args[0]
        if thread_id is not None:
            span.set_attribute("gen_ai.session.id", str(thread_id))
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            span.end()
        return result

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        method_name = getattr(wrapped, "__name__", "acquire")
        return self._run(wrapped, instance, args, kwargs, method_name=method_name)


class _ProviderAcquireAsyncWrapper(_ProviderLifecycleWrapper):
    async def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        method_name = getattr(wrapped, "__name__", "acquire_async")
        # Run sync path in a thread so the span covers the actual work.
        import asyncio

        return await asyncio.to_thread(
            self._run, wrapped, instance, args, kwargs, method_name=method_name
        )


def instrument(handler: ExtendedTelemetryHandler) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []

    # Sandbox.execute_command / read_file / write_file / glob / grep / list_dir
    for method_name in _SANDBOX_METHODS:
        attr = f"{_SANDBOX_CLASS}.{method_name}"
        try:
            wrap_function_wrapper(
                _SANDBOX_MODULE, attr, _SandboxMethodWrapper(handler)
            )
            targets.append((_SANDBOX_MODULE, attr))
        except Exception as exc:
            logger.debug(
                "DeerFlow: could not wrap %s.%s: %s",
                _SANDBOX_MODULE,
                attr,
                exc,
            )

    # SandboxProvider.acquire / acquire_async / release
    tracer = get_tracer("opentelemetry.instrumentation.deer_flow")
    for method_name, wrapper in (
        ("acquire", _ProviderLifecycleWrapper(tracer)),
        ("acquire_async", _ProviderAcquireAsyncWrapper(tracer)),
        ("release", _ProviderLifecycleWrapper(tracer)),
    ):
        attr = f"{_PROVIDER_CLASS}.{method_name}"
        try:
            wrap_function_wrapper(_PROVIDER_MODULE, attr, wrapper)
            targets.append((_PROVIDER_MODULE, attr))
        except Exception as exc:
            logger.debug(
                "DeerFlow: could not wrap %s.%s: %s",
                _PROVIDER_MODULE,
                attr,
                exc,
            )

    return targets
