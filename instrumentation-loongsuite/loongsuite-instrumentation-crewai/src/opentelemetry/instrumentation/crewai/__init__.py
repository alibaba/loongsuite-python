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

"""OpenTelemetry instrumentation for CrewAI."""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from importlib import import_module
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.crewai.package import _instruments
from opentelemetry.instrumentation.crewai.utils import (
    OP_NAME_CREW,
    OP_NAME_FLOW,
    GenAIHookHelper,
    apply_usage_metrics,
    create_agent_invocation,
    create_entry_invocation,
    create_task_invocation,
    create_tool_invocation,
    to_output_messages,
    usage_metric_attributes,
)
from opentelemetry.instrumentation.crewai.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)
_ENTRY_DEPTH: ContextVar[int] = ContextVar("crewai_entry_depth", default=0)
_TASK_DEPTH: ContextVar[int] = ContextVar("crewai_task_depth", default=0)
_CREWAI_UNINSTRUMENT_TARGETS = (
    ("crewai.crew", "Crew", "kickoff"),
    ("crewai.crew", "Crew", "kickoff_async"),
    ("crewai.flow.flow", "Flow", "kickoff"),
    ("crewai.flow.flow", "Flow", "kickoff_async"),
    ("crewai.agent", "Agent", "execute_task"),
    ("crewai.task", "Task", "execute_sync"),
    ("crewai.tools.tool_usage", "ToolUsage", "_use"),
)


def _set_ok(invocation: Any) -> None:
    span = getattr(invocation, "span", None)
    if span is not None:
        span.set_status(Status(StatusCode.OK))


def _record_exception(invocation: Any, exc: BaseException) -> None:
    span = getattr(invocation, "span", None)
    if span is not None and span.is_recording():
        span.record_exception(exc)


def _error(exc: BaseException) -> Error:
    return Error(message=str(exc) or type(exc).__name__, type=type(exc))


def _safe_handler_call(action: str, method: Any, *args: Any) -> bool:
    try:
        method(*args)
        return True
    except Exception as exc:
        logger.warning(
            "CrewAI instrumentation handler %s failed: %s",
            action,
            exc,
            exc_info=True,
        )
        return False


def _safe_build_invocation(
    action: str, factory: Any, *args: Any, **kwargs: Any
) -> Any:
    try:
        return factory(*args, **kwargs)
    except Exception as exc:
        logger.warning(
            "CrewAI instrumentation %s invocation build failed: %s",
            action,
            exc,
            exc_info=True,
        )
        return None


def _safe_post_process(action: str, callback: Any) -> None:
    try:
        callback()
    except Exception as exc:
        logger.warning(
            "CrewAI instrumentation %s post-processing failed: %s",
            action,
            exc,
            exc_info=True,
        )


def _uninstrument_targets() -> list[tuple[Any, str]]:
    targets = []
    for module_name, class_name, method_name in _CREWAI_UNINSTRUMENT_TARGETS:
        try:
            module = import_module(module_name)
            target = getattr(module, class_name)
        except ImportError:
            logger.debug(
                "CrewAI module %s was not available for uninstrumentation.",
                module_name,
                exc_info=True,
            )
            continue
        except Exception as exc:
            logger.warning(
                "Could not resolve CrewAI target %s.%s for "
                "uninstrumentation: %s",
                module_name,
                class_name,
                exc,
                exc_info=True,
            )
            continue
        targets.append((target, method_name))
    return targets


def _input_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "inputs" in kwargs:
        inputs = kwargs["inputs"]
        return {} if inputs is None else inputs
    if args and (args[0] is None or isinstance(args[0], dict)):
        return {} if args[0] is None else args[0]
    return {}


def _call_arg(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    index: int,
    name: str,
    default: Any = None,
) -> Any:
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _looks_like_tool(value: Any) -> bool:
    return bool(getattr(value, "name", None)) and bool(
        getattr(value, "description", None)
    )


def _looks_like_tool_calling(value: Any) -> bool:
    if isinstance(value, dict):
        return "arguments" in value or "tool_input" in value
    return (
        getattr(value, "arguments", None) is not None
        or getattr(value, "tool_name", None) is not None
        or getattr(value, "tool_call_id", None) is not None
    )


def _tool_call_from_call(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, Any]:
    tool = kwargs.get("tool")
    calling = kwargs.get("calling") or kwargs.get("tool_calling")

    if tool is None:
        for candidate in args:
            if _looks_like_tool(candidate):
                tool = candidate
                break
    if calling is None:
        for candidate in args:
            if candidate is tool:
                continue
            if _looks_like_tool_calling(candidate):
                calling = candidate
                break

    return tool or _call_arg(args, kwargs, 0, "tool"), calling


def _tool_call_arguments(instance: Any, calling: Any) -> Any:
    arguments = getattr(calling, "arguments", None)
    if arguments is None and isinstance(calling, dict):
        arguments = calling.get("arguments") or calling.get("tool_input")
    if arguments is not None:
        return arguments

    action = getattr(instance, "action", None)
    if action is None:
        return None
    return getattr(action, "tool_input", None)


def _agent_usage_sources(agent: Any) -> tuple[Any, ...]:
    if agent is None:
        return ()
    return (
        getattr(agent, "_token_process", None),
        getattr(agent, "llm", None),
    )


def _crew_usage_sources(instance: Any) -> tuple[Any, ...]:
    agents = getattr(instance, "agents", None)
    if not isinstance(agents, (list, tuple)):
        return ()
    sources: list[Any] = []
    for agent in agents:
        sources.extend(_agent_usage_sources(agent))
    return tuple(sources)


def _should_skip_entry(instance: Any) -> bool:
    return _ENTRY_DEPTH.get() > 0 or getattr(instance, "stream", False) is True


def _enter_entry() -> Token[int]:
    return _ENTRY_DEPTH.set(_ENTRY_DEPTH.get() + 1)


def _enter_task() -> Token[int]:
    return _TASK_DEPTH.set(_TASK_DEPTH.get() + 1)


def _result_content(result: Any) -> Any:
    if isinstance(result, (str, bytes, bytearray)):
        return result
    if hasattr(result, "result"):
        try:
            return result.result
        except Exception:
            pass
    return result


def _finish_entry_success(
    handler: ExtendedTelemetryHandler,
    invocation: Any,
    started: bool,
    result: Any,
    instance: Any,
) -> None:
    def post_process() -> None:
        invocation.output_messages = to_output_messages(
            "assistant", _result_content(result)
        )
        invocation.attributes.update(
            usage_metric_attributes(
                result, instance, *_crew_usage_sources(instance)
            )
        )
        _set_ok(invocation)

    _safe_post_process("entry success", post_process)
    if started:
        _safe_handler_call("stop_entry", handler.stop_entry, invocation)


def _fail_entry(
    handler: ExtendedTelemetryHandler,
    invocation: Any,
    started: bool,
    exc: BaseException,
) -> None:
    if not started:
        return
    _record_exception(invocation, exc)
    _safe_handler_call(
        "fail_entry",
        handler.fail_entry,
        invocation,
        _error(exc),
    )


def _finish_agent_success(
    handler: ExtendedTelemetryHandler,
    invocation: Any,
    started: bool,
    result: Any,
    *usage_sources: Any,
) -> None:
    def post_process() -> None:
        invocation.output_messages = to_output_messages(
            "assistant", _result_content(result)
        )
        apply_usage_metrics(invocation, result, *usage_sources)
        _set_ok(invocation)

    _safe_post_process("agent success", post_process)
    if started:
        _safe_handler_call(
            "stop_invoke_agent",
            handler.stop_invoke_agent,
            invocation,
        )


def _fail_agent(
    handler: ExtendedTelemetryHandler,
    invocation: Any,
    started: bool,
    exc: BaseException,
) -> None:
    if not started:
        return
    _record_exception(invocation, exc)
    _safe_handler_call(
        "fail_invoke_agent",
        handler.fail_invoke_agent,
        invocation,
        _error(exc),
    )


def _is_tool_parsing_error(instance: Any) -> bool:
    if not instance:
        return False
    run_attempts = getattr(instance, "_run_attempts", None)
    max_parsing_attempts = getattr(instance, "_max_parsing_attempts", None)
    return bool(
        max_parsing_attempts
        and run_attempts
        and run_attempts > max_parsing_attempts
    )


def _finish_tool_success(
    handler: ExtendedTelemetryHandler,
    invocation: Any,
    started: bool,
    result: Any,
    instance: Any,
    tool: Any,
) -> None:
    error: RuntimeError | None = None

    def post_process() -> None:
        nonlocal error
        invocation.tool_call_result = result
        if _is_tool_parsing_error(instance):
            error = RuntimeError(
                "CrewAI tool parsing attempts exceeded for "
                f"{getattr(tool, 'name', 'unknown_tool')}: "
                f"{getattr(instance, '_run_attempts', None)}/"
                f"{getattr(instance, '_max_parsing_attempts', None)}"
            )
            _record_exception(invocation, error)
        else:
            _set_ok(invocation)

    _safe_post_process("tool success", post_process)
    if not started:
        return
    if error is not None:
        _safe_handler_call(
            "fail_execute_tool",
            handler.fail_execute_tool,
            invocation,
            _error(error),
        )
    else:
        _safe_handler_call(
            "stop_execute_tool",
            handler.stop_execute_tool,
            invocation,
        )


def _fail_tool(
    handler: ExtendedTelemetryHandler,
    invocation: Any,
    started: bool,
    exc: BaseException,
) -> None:
    if not started:
        return
    _record_exception(invocation, exc)
    _safe_handler_call(
        "fail_execute_tool",
        handler.fail_execute_tool,
        invocation,
        _error(exc),
    )


def _wrap_stream_result(
    result: Any,
    on_success: Any,
    on_error: Any,
    cleanup: Any,
) -> bool:
    sync_iterator = getattr(result, "_sync_iterator", None)
    async_iterator = getattr(result, "_async_iterator", None)

    if sync_iterator is not None:

        def iterate():
            try:
                for chunk in sync_iterator:
                    yield chunk
            except Exception as exc:
                on_error(exc)
                raise
            else:
                on_success(result)
            finally:
                cleanup()

        result._sync_iterator = iterate()
        return True

    if async_iterator is not None:

        async def async_iterate():
            try:
                async for chunk in async_iterator:
                    yield chunk
            except Exception as exc:
                on_error(exc)
                raise
            else:
                on_success(result)
            finally:
                cleanup()

        result._async_iterator = async_iterate()
        return True

    return False


def _run_entry_sync(
    handler: ExtendedTelemetryHandler,
    operation_name: str,
    wrapped: Any,
    instance: Any,
    args: Any,
    kwargs: Any,
) -> Any:
    if _should_skip_entry(instance):
        return wrapped(*args, **kwargs)

    def build_invocation() -> Any:
        return create_entry_invocation(
            instance,
            _input_from_call(args, kwargs),
            operation_name,
        )

    invocation = _safe_build_invocation(operation_name, build_invocation)
    if invocation is None:
        return wrapped(*args, **kwargs)

    token = _enter_entry()
    started = _safe_handler_call(
        "start_entry", handler.start_entry, invocation
    )
    cleanup_done = False

    def cleanup() -> None:
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        _ENTRY_DEPTH.reset(token)

    try:
        result = wrapped(*args, **kwargs)
    except Exception as exc:
        try:
            _fail_entry(handler, invocation, started, exc)
        finally:
            cleanup()
        raise

    if _wrap_stream_result(
        result,
        lambda stream_result: _finish_entry_success(
            handler, invocation, started, stream_result, instance
        ),
        lambda exc: _fail_entry(handler, invocation, started, exc),
        cleanup,
    ):
        return result

    try:
        _finish_entry_success(handler, invocation, started, result, instance)
        return result
    finally:
        cleanup()


async def _run_entry_async(
    handler: ExtendedTelemetryHandler,
    operation_name: str,
    wrapped: Any,
    instance: Any,
    args: Any,
    kwargs: Any,
) -> Any:
    if _should_skip_entry(instance):
        return await wrapped(*args, **kwargs)

    def build_invocation() -> Any:
        return create_entry_invocation(
            instance,
            _input_from_call(args, kwargs),
            operation_name,
        )

    invocation = _safe_build_invocation(operation_name, build_invocation)
    if invocation is None:
        return await wrapped(*args, **kwargs)

    token = _enter_entry()
    started = _safe_handler_call(
        "start_entry", handler.start_entry, invocation
    )
    cleanup_done = False

    def cleanup() -> None:
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        _ENTRY_DEPTH.reset(token)

    try:
        result = await wrapped(*args, **kwargs)
    except Exception as exc:
        try:
            _fail_entry(handler, invocation, started, exc)
        finally:
            cleanup()
        raise

    if _wrap_stream_result(
        result,
        lambda stream_result: _finish_entry_success(
            handler, invocation, started, stream_result, instance
        ),
        lambda exc: _fail_entry(handler, invocation, started, exc),
        cleanup,
    ):
        return result

    try:
        _finish_entry_success(handler, invocation, started, result, instance)
        return result
    finally:
        cleanup()


class CrewAIInstrumentor(BaseInstrumentor):
    """Instrumentor for the CrewAI framework."""

    def __init__(self) -> None:
        super().__init__()
        self._handler: ExtendedTelemetryHandler | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        self._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        for module, name, wrapper in (
            (
                "crewai.crew",
                "Crew.kickoff",
                _CrewKickoffWrapper(self._handler),
            ),
            (
                "crewai.crew",
                "Crew.kickoff_async",
                _CrewKickoffAsyncWrapper(self._handler),
            ),
            (
                "crewai.flow.flow",
                "Flow.kickoff",
                _FlowKickoffWrapper(self._handler),
            ),
            (
                "crewai.flow.flow",
                "Flow.kickoff_async",
                _FlowKickoffAsyncWrapper(self._handler),
            ),
            (
                "crewai.agent",
                "Agent.execute_task",
                _AgentExecuteTaskWrapper(self._handler),
            ),
            (
                "crewai.task",
                "Task.execute_sync",
                _TaskExecuteSyncWrapper(self._handler),
            ),
            (
                "crewai.tools.tool_usage",
                "ToolUsage._use",
                _ToolUseWrapper(self._handler),
            ),
        ):
            try:
                wrap_function_wrapper(
                    module=module, name=name, wrapper=wrapper
                )
            except Exception as exc:
                logger.warning("Could not wrap %s: %s", name, exc)

    def _uninstrument(self, **kwargs: Any) -> None:
        del kwargs
        for target, method_name in _uninstrument_targets():
            try:
                unwrap(target, method_name)
            except Exception as exc:
                logger.debug(
                    "Could not unwrap %s.%s: %s",
                    getattr(target, "__name__", target),
                    method_name,
                    exc,
                )

        self._handler = None


class _CrewKickoffWrapper:
    """Wrap ``Crew.kickoff`` as a util-genai Entry span."""

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any):
        return _run_entry_sync(
            self._handler, OP_NAME_CREW, wrapped, instance, args, kwargs
        )


class _CrewKickoffAsyncWrapper:
    """Wrap ``Crew.kickoff_async`` as a util-genai Entry span."""

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    async def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ):
        return await _run_entry_async(
            self._handler, OP_NAME_CREW, wrapped, instance, args, kwargs
        )


class _FlowKickoffWrapper:
    """Wrap ``Flow.kickoff`` as a util-genai Entry span."""

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any):
        return _run_entry_sync(
            self._handler, OP_NAME_FLOW, wrapped, instance, args, kwargs
        )


class _FlowKickoffAsyncWrapper:
    """Wrap ``Flow.kickoff_async`` as a util-genai Entry span."""

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    async def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ):
        return await _run_entry_async(
            self._handler, OP_NAME_FLOW, wrapped, instance, args, kwargs
        )


class _AgentExecuteTaskWrapper:
    """Wrap ``Agent.execute_task`` as a util-genai invoke_agent span."""

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any):
        if _TASK_DEPTH.get() > 0:
            return wrapped(*args, **kwargs)

        task = _call_arg(args, kwargs, 0, "task")
        context = _call_arg(args, kwargs, 1, "context", "")
        tools = _call_arg(args, kwargs, 2, "tools", [])
        invocation = _safe_build_invocation(
            "agent",
            create_agent_invocation,
            instance,
            task,
            context,
            tools,
        )
        if invocation is None:
            return wrapped(*args, **kwargs)

        started = _safe_handler_call(
            "start_invoke_agent", self._handler.start_invoke_agent, invocation
        )
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            _fail_agent(self._handler, invocation, started, exc)
            raise

        _finish_agent_success(
            self._handler,
            invocation,
            started,
            result,
            instance,
            *_agent_usage_sources(instance),
            getattr(instance, "crew", None),
        )
        return result


class _TaskExecuteSyncWrapper:
    """Wrap ``Task.execute_sync`` as a util-genai invoke_agent span."""

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any):
        agent = _call_arg(args, kwargs, 0, "agent")
        if agent is None:
            agent = getattr(instance, "agent", None)
        invocation = _safe_build_invocation(
            "task", create_task_invocation, instance, agent
        )
        if invocation is None:
            return wrapped(*args, **kwargs)

        started = _safe_handler_call(
            "start_invoke_agent", self._handler.start_invoke_agent, invocation
        )
        task_token = _enter_task()
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            _fail_agent(self._handler, invocation, started, exc)
            raise
        finally:
            _TASK_DEPTH.reset(task_token)

        _finish_agent_success(
            self._handler,
            invocation,
            started,
            result,
            instance,
            agent,
            *_agent_usage_sources(agent),
            getattr(agent, "crew", None),
        )
        return result


class _ToolUseWrapper:
    """Wrap ``ToolUsage._use`` as a util-genai execute_tool span."""

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any):
        try:
            tool, tool_calling = _tool_call_from_call(args, kwargs)
            tool_call_arguments = _tool_call_arguments(instance, tool_calling)
        except Exception as exc:
            logger.warning(
                "CrewAI instrumentation tool invocation build failed: %s",
                exc,
                exc_info=True,
            )
            return wrapped(*args, **kwargs)

        invocation = _safe_build_invocation(
            "tool",
            create_tool_invocation,
            tool,
            tool_calling,
            tool_call_arguments=tool_call_arguments,
        )
        if invocation is None:
            return wrapped(*args, **kwargs)

        started = _safe_handler_call(
            "start_execute_tool", self._handler.start_execute_tool, invocation
        )
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            _fail_tool(self._handler, invocation, started, exc)
            raise

        _finish_tool_success(
            self._handler,
            invocation,
            started,
            result,
            instance,
            tool,
        )
        return result


__all__ = [
    "__version__",
    "CrewAIInstrumentor",
    "GenAIHookHelper",
    "_CrewKickoffWrapper",
    "_CrewKickoffAsyncWrapper",
    "_FlowKickoffWrapper",
    "_FlowKickoffAsyncWrapper",
    "_AgentExecuteTaskWrapper",
    "_TaskExecuteSyncWrapper",
    "_ToolUseWrapper",
]
