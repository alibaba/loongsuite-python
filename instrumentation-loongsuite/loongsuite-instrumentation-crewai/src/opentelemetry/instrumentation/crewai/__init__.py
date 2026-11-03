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
# The tool whose execute_tool span is open in this context, if any.
_ACTIVE_TOOL: ContextVar[Any] = ContextVar("crewai_active_tool", default=None)
# The model's id of the native tool call that CrewAI runs in this context.
_NATIVE_TOOL_CALL_ID: ContextVar[str | None] = ContextVar(
    "crewai_native_tool_call_id", default=None
)
# Where CrewAI runs one native tool call; each takes the model's tool call
# first. Missing ones (other CrewAI versions) are skipped, and the spans of
# their tool calls fall back to the tool name as the call id.
_NATIVE_TOOL_CALL_TARGETS = (
    (
        "crewai.agents.agent_executor",
        "AgentExecutor",
        "_execute_single_native_tool_call",
    ),
    ("crewai.agents.step_executor", None, "execute_single_native_tool_call"),
)
_CREWAI_UNINSTRUMENT_TARGETS = (
    ("crewai.crew", "Crew", "kickoff"),
    ("crewai.crew", "Crew", "kickoff_async"),
    ("crewai.flow.flow", "Flow", "kickoff"),
    ("crewai.flow.flow", "Flow", "kickoff_async"),
    ("crewai.agent", "Agent", "execute_task"),
    ("crewai.task", "Task", "execute_sync"),
    ("crewai.tools.tool_usage", "ToolUsage", "_use"),
    # BaseTool first: unwrapping Tool.run, when Tool inherits it, is then a
    # no-op.
    ("crewai.tools.base_tool", "BaseTool", "run"),
    ("crewai.tools.base_tool", "Tool", "run"),
    *_NATIVE_TOOL_CALL_TARGETS,
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
            target = (
                module if class_name is None else getattr(module, class_name)
            )
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
        invocation.tool_call_result = _tool_result_for_agent(tool, result)
        if _is_tool_parsing_error(instance):
            error = RuntimeError(
                "CrewAI tool parsing attempts exceeded for "
                f"{getattr(tool, 'name', 'unknown_tool')}: "
                f"{getattr(instance, '_run_attempts', None)}/"
                f"{getattr(instance, '_max_parsing_attempts', None)}"
            )
            _record_exception(invocation, error)
            return
        failure = _returned_tool_failure(result, instance)
        if failure is None:
            _set_ok(invocation)
        else:
            # Returned, not raised: an error status but no exception event.
            error = RuntimeError(failure.as_agent_message())

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


def _tool_result_for_agent(tool: Any, result: Any) -> Any:
    """``result`` as the text the agent gets, as ``ToolUsage._use`` returns.

    ``BaseTool.run`` returns the raw result, which need not be JSON
    serializable: a pydantic model or a ``ToolFailure``, for instance.
    """
    if result is None or isinstance(result, str):
        return result
    format_output = getattr(tool, "format_output_for_agent", None)
    if callable(format_output):
        try:
            return format_output(result)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.debug(
                "Could not format a CrewAI tool result", exc_info=True
            )
    return str(result)


def _returned_tool_failure(result: Any, instance: Any) -> Any:
    """The ``ToolFailure`` that the tool returned instead of raising, if any.

    ``ToolUsage._use`` keeps it in ``last_failure`` and returns the text the
    agent gets; ``BaseTool.run`` returns it as is.
    """
    try:
        tool_failure_type = import_module(
            "crewai.tools.tool_failure"
        ).ToolFailure
    except (ImportError, AttributeError):
        return None
    for candidate in (getattr(instance, "last_failure", None), result):
        if isinstance(candidate, tool_failure_type):
            return candidate
    return None


def _native_tool_call_id(tool_call: Any) -> str | None:
    """The model's id of a native tool call, as CrewAI itself reads it."""
    try:
        info = import_module(
            "crewai.utilities.agent_utils"
        ).extract_tool_call_info(tool_call)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    return info[0] if info and isinstance(info[0], str) else None


def _wrap_native_tool_call(
    wrapped: Any, instance: Any, args: Any, kwargs: Any
) -> Any:
    """Keep the model's call id while CrewAI runs one native tool call."""
    del instance
    tool_call = args[0] if args else kwargs.get("tool_call")
    token = _NATIVE_TOOL_CALL_ID.set(_native_tool_call_id(tool_call))
    try:
        return wrapped(*args, **kwargs)
    finally:
        _NATIVE_TOOL_CALL_ID.reset(token)


def _run_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The arguments of a ``BaseTool.run`` call, as recorded on its span."""
    if args and kwargs:
        return {"args": list(args), "kwargs": kwargs}
    if args:
        return list(args)
    return kwargs or None


def _is_active_tool(tool: Any) -> bool:
    """Whether an execute_tool span for ``tool`` is already open here.

    ``ToolUsage._use`` gets the ``CrewStructuredTool`` built from a
    ``BaseTool``, so the structured tool's ``_original_tool`` counts too.
    """
    active = _ACTIVE_TOOL.get()
    return active is not None and (
        active is tool or getattr(active, "_original_tool", None) is tool
    )


def _tool_class_overrides_run() -> bool:
    """Whether crewAI's ``Tool`` (behind ``@tool``) has its own ``run``.

    Older releases inherit ``BaseTool.run``; wrapping it again there would
    leave a stale wrapper behind on uninstrumentation.
    """
    try:
        tool_class = import_module("crewai.tools.base_tool").Tool
    except (ImportError, AttributeError):
        return False
    return "run" in vars(tool_class)


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

        wrappers = [
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
            # Native function calls run tools through ``run``, not ``_use``.
            (
                "crewai.tools.base_tool",
                "BaseTool.run",
                _ToolRunWrapper(self._handler),
            ),
        ]
        if _tool_class_overrides_run():
            wrappers.append(
                (
                    "crewai.tools.base_tool",
                    "Tool.run",
                    _ToolRunWrapper(self._handler),
                )
            )
        for module, name, wrapper in wrappers:
            try:
                wrap_function_wrapper(
                    module=module, name=name, wrapper=wrapper
                )
            except Exception as exc:
                logger.warning("Could not wrap %s: %s", name, exc)

        for module, class_name, method_name in _NATIVE_TOOL_CALL_TARGETS:
            name = (
                method_name
                if class_name is None
                else f"{class_name}.{method_name}"
            )
            try:
                wrap_function_wrapper(
                    module=module, name=name, wrapper=_wrap_native_tool_call
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Could not wrap %s.%s: %s", module, name, exc)

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
        token = _ACTIVE_TOOL.set(tool)
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            _fail_tool(self._handler, invocation, started, exc)
            raise
        finally:
            _ACTIVE_TOOL.reset(token)

        _finish_tool_success(
            self._handler,
            invocation,
            started,
            result,
            instance,
            tool,
        )
        return result


class _ToolRunWrapper:
    """Wrap ``BaseTool.run`` as a util-genai execute_tool span.

    With native function calling, crewAI runs tools through ``run`` instead
    of ``ToolUsage._use``. The span takes the model's call id from the native
    tool call being run, if any. A ``run`` inside an already traced
    execution of the same tool gets no second span.
    """

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any):
        if _is_active_tool(instance):
            return wrapped(*args, **kwargs)

        call_id = _NATIVE_TOOL_CALL_ID.get()
        invocation = _safe_build_invocation(
            "tool",
            create_tool_invocation,
            instance,
            {"id": call_id} if call_id else None,
            tool_call_arguments=_run_arguments(args, kwargs),
        )
        if invocation is None:
            return wrapped(*args, **kwargs)

        started = _safe_handler_call(
            "start_execute_tool", self._handler.start_execute_tool, invocation
        )
        tool_token = _ACTIVE_TOOL.set(instance)
        # A tool run from inside this one is not the model's tool call.
        call_id_token = _NATIVE_TOOL_CALL_ID.set(None)
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            _fail_tool(self._handler, invocation, started, exc)
            raise
        finally:
            _NATIVE_TOOL_CALL_ID.reset(call_id_token)
            _ACTIVE_TOOL.reset(tool_token)

        _finish_tool_success(
            self._handler, invocation, started, result, None, instance
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
    "_ToolRunWrapper",
    "_ToolUseWrapper",
]
