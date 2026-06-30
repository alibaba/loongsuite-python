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

from __future__ import annotations

import asyncio
import logging
import timeit
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_span_utils import (
    _apply_invoke_agent_finish_attributes,
)
from opentelemetry.util.genai.types import Error, InputMessage, Text

from .config import is_agent_span_enabled, is_llm_span_enabled
from .semantic_conventions import (
    AUTOGEN_PROVIDER_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_SPAN_KIND,
    GEN_AI_SYSTEM,
    GenAIOperation,
    GenAISpanKind,
)
from .utils import (
    apply_agent_input,
    apply_agent_stream_item,
    apply_create_result,
    apply_tool_result,
    make_agent_invocation,
    make_llm_invocation,
    make_tool_invocation,
)

logger = logging.getLogger(__name__)

_ASSISTANT_AGENT_MODULE = "autogen_agentchat.agents._assistant_agent"
_CODE_EXECUTOR_AGENT_MODULE = "autogen_agentchat.agents._code_executor_agent"
_SOCIETY_OF_MIND_AGENT_MODULE = (
    "autogen_agentchat.agents._society_of_mind_agent"
)
_USER_PROXY_AGENT_MODULE = "autogen_agentchat.agents._user_proxy_agent"
_SELECTOR_GROUP_CHAT_MODULE = (
    "autogen_agentchat.teams._group_chat._selector_group_chat"
)
_BASE_GROUP_CHAT_MODULE = (
    "autogen_agentchat.teams._group_chat._base_group_chat"
)
_BASE_TOOL_MODULE = "autogen_core.tools._base"
_applied = False
_optional_wrappers: list[tuple[str, str]] = []
_suppress_native_tool_span: ContextVar[bool] = ContextVar(
    "loongsuite_autogen_suppress_native_tool_span", default=False
)
_suppress_direct_model_span: ContextVar[bool] = ContextVar(
    "loongsuite_autogen_suppress_direct_model_span", default=False
)
_direct_model_agent_name: ContextVar[str | None] = ContextVar(
    "loongsuite_autogen_direct_model_agent_name", default=None
)

_CALL_LLM_PARAM_NAMES = (
    "model_client",
    "model_client_stream",
    "system_messages",
    "model_context",
    "workbench",
    "handoff_tools",
    "agent_name",
    "cancellation_token",
    "output_content_type",
    "message_id",
)

_ON_MESSAGES_PARAM_NAMES = (
    "messages",
    "cancellation_token",
)

_EXECUTE_TOOL_CALL_PARAM_NAMES = (
    "tool_call",
    "workbench",
    "handoff_tools",
    "agent_name",
    "cancellation_token",
    "stream",
)

_MODEL_CLIENT_CREATE_PARAM_NAMES = (
    "messages",
    "tools",
    "tool_choice",
    "json_output",
    "extra_create_args",
    "cancellation_token",
)

# Patch model clients that can execute model requests directly. Cache and
# replay wrappers are intentionally excluded to avoid duplicate or fake LLM spans.
_DIRECT_MODEL_CLIENT_PATCHES = (
    (
        "autogen_ext.models.openai._openai_client",
        "BaseOpenAIChatCompletionClient",
    ),
    (
        "autogen_ext.models.anthropic._anthropic_client",
        "BaseAnthropicChatCompletionClient",
    ),
    (
        "autogen_ext.models.ollama._ollama_client",
        "BaseOllamaChatCompletionClient",
    ),
    (
        "autogen_ext.models.azure._azure_ai_client",
        "AzureAIChatCompletionClient",
    ),
    (
        "autogen_ext.models.semantic_kernel._sk_chat_completion_adapter",
        "SKChatCompletionAdapter",
    ),
    (
        "autogen_ext.models.llama_cpp._llama_cpp_completion_client",
        "LlamaCppChatCompletionClient",
    ),
)


def _span_attr(span: Any, key: str) -> Any:
    attrs = getattr(span, "_attributes", None)
    if attrs is not None:
        try:
            return attrs.get(key)
        except Exception:
            pass
    attrs = getattr(span, "attributes", None)
    if attrs is not None:
        try:
            return attrs.get(key)
        except Exception:
            pass
    return None


def _current_autogen_agent_span_active() -> bool:
    return _current_autogen_agent_span() is not None


def _current_autogen_agent_span() -> Any | None:
    span = trace.get_current_span()
    if span is None:
        return None
    operation = _span_attr(span, GEN_AI_OPERATION_NAME)
    provider = _span_attr(span, GEN_AI_PROVIDER_NAME)
    system = _span_attr(span, GEN_AI_SYSTEM)
    span_kind = _span_attr(span, GEN_AI_SPAN_KIND)
    if (
        operation == GenAIOperation.INVOKE_AGENT
        and (
            provider == AUTOGEN_PROVIDER_NAME
            or system == AUTOGEN_PROVIDER_NAME
        )
        and (span_kind in (None, GenAISpanKind.AGENT))
    ):
        return span
    return None


def _arg_value(
    args: tuple[Any, ...], kwargs: dict[str, Any], name: str
) -> Any:
    if name in kwargs:
        return kwargs[name]
    idx = _CALL_LLM_PARAM_NAMES.index(name)
    # wrapt may include the class object as args[0] for classmethod wrappers on
    # some Python/wrapt combinations. Handle both shapes.
    if args and isinstance(args[0], type):
        idx += 1
    if idx < len(args):
        return args[idx]
    return None


def _named_arg_value(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    names: tuple[str, ...],
) -> Any:
    if name in kwargs:
        return kwargs[name]
    idx = names.index(name)
    if args and isinstance(args[0], type):
        idx += 1
    if idx < len(args):
        return args[idx]
    return None


def _error_from_exception(exc: BaseException) -> Error:
    message = str(exc) or type(exc).__name__
    return Error(message=message, type=type(exc))


def _json_output_type(value: Any) -> str | None:
    return "json" if value else None


def _apply_invoke_agent_attrs(
    invocation: Any, span: Any | None = None
) -> None:
    span = span or trace.get_current_span()
    if span is None:
        return
    is_recording = getattr(span, "is_recording", None)
    if callable(is_recording) and not is_recording():
        return
    try:
        _apply_invoke_agent_finish_attributes(span, invocation)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("AutoGen invoke_agent span enrichment failed: %s", exc)


def _model_client_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any], name: str
) -> Any:
    return _named_arg_value(
        args, kwargs, name, _MODEL_CLIENT_CREATE_PARAM_NAMES
    )


def _make_direct_model_invocation(
    instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    agent_name = _direct_model_agent_name.get()
    return make_llm_invocation(
        instance,
        _model_client_arg(args, kwargs, "messages"),
        _model_client_arg(args, kwargs, "tools") or [],
        agent_name=agent_name,
        output_type=_json_output_type(
            _model_client_arg(args, kwargs, "json_output")
        ),
    )


def _apply_team_input(
    invocation: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    task = kwargs.get("task")
    if task is None and args:
        task = args[0]
    if task is None:
        return
    if isinstance(task, str):
        invocation.input_messages.append(
            InputMessage(role="user", parts=[Text(content=task)])
        )
    elif isinstance(task, (list, tuple)):
        apply_agent_input(invocation, list(task))
    else:
        apply_agent_input(invocation, [task])


def _apply_team_attributes(invocation: Any, instance: Any) -> None:
    invocation.attributes["autogen.team.type"] = type(instance).__name__
    participants = getattr(instance, "_participant_names", None)
    if participants:
        invocation.attributes["autogen.team.participants"] = [
            str(name) for name in participants
        ]


async def _contextual_async_iter(
    async_iterable: Any, context_var: ContextVar[Any], value: Any
):  # type: ignore[no-untyped-def]
    iterator = async_iterable.__aiter__()
    try:
        while True:
            token = context_var.set(value)
            try:
                item = await iterator.__anext__()
            except StopAsyncIteration:
                return
            finally:
                context_var.reset(token)
            yield item
    finally:
        close = getattr(iterator, "aclose", None)
        if callable(close):
            await close()


async def _collect_llm_messages(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[Any]:
    system_messages = _arg_value(args, kwargs, "system_messages") or []
    model_context = _arg_value(args, kwargs, "model_context")
    context_messages: list[Any] = []
    get_messages = getattr(model_context, "get_messages", None)
    if callable(get_messages):
        try:
            context_messages = list(await get_messages())
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("AutoGen model_context.get_messages failed: %s", exc)
    return list(system_messages) + context_messages


async def _collect_tools(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[Any]:
    tools: list[Any] = []
    workbench = _arg_value(args, kwargs, "workbench") or []
    for wb in workbench:
        list_tools = getattr(wb, "list_tools", None)
        if callable(list_tools):
            try:
                tools.extend(await list_tools())
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("AutoGen workbench.list_tools failed: %s", exc)
    tools.extend(_arg_value(args, kwargs, "handoff_tools") or [])
    return tools


def _on_messages_stream_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
    if not is_agent_span_enabled(default=True):
        return wrapped(*args, **kwargs)

    async def _generator():  # type: ignore[no-untyped-def]
        handler = _get_handler()
        invocation = make_agent_invocation(instance)
        apply_agent_input(
            invocation,
            _named_arg_value(
                args, kwargs, "messages", _ON_MESSAGES_PARAM_NAMES
            ),
        )
        native_agent_span = _current_autogen_agent_span()
        if native_agent_span is None:
            handler.start_invoke_agent(invocation)
        else:
            _apply_invoke_agent_attrs(invocation, native_agent_span)
        try:
            async for item in _contextual_async_iter(
                wrapped(*args, **kwargs),
                _direct_model_agent_name,
                invocation.agent_name,
            ):
                apply_agent_stream_item(
                    invocation, item, timeit.default_timer()
                )
                if (
                    native_agent_span is not None
                    and type(item).__name__ == "Response"
                ):
                    _apply_invoke_agent_attrs(invocation, native_agent_span)
                yield item
        except (GeneratorExit, asyncio.CancelledError) as exc:
            if native_agent_span is None:
                handler.fail_invoke_agent(
                    invocation, _error_from_exception(exc)
                )
            else:
                _apply_invoke_agent_attrs(invocation, native_agent_span)
            raise
        except Exception as exc:
            if native_agent_span is None:
                handler.fail_invoke_agent(
                    invocation, _error_from_exception(exc)
                )
            else:
                _apply_invoke_agent_attrs(invocation, native_agent_span)
            raise
        if native_agent_span is None:
            handler.stop_invoke_agent(invocation)
        else:
            _apply_invoke_agent_attrs(invocation, native_agent_span)

    return _generator()


def _team_run_stream_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
    if not is_agent_span_enabled(default=True):
        return wrapped(*args, **kwargs)

    async def _generator():  # type: ignore[no-untyped-def]
        handler = _get_handler()
        invocation = make_agent_invocation(instance)
        _apply_team_input(invocation, args, kwargs)
        _apply_team_attributes(invocation, instance)
        handler.start_invoke_agent(invocation)
        try:
            async for item in wrapped(*args, **kwargs):
                apply_agent_stream_item(
                    invocation, item, timeit.default_timer()
                )
                yield item
        except (GeneratorExit, asyncio.CancelledError) as exc:
            handler.fail_invoke_agent(invocation, _error_from_exception(exc))
            raise
        except Exception as exc:
            handler.fail_invoke_agent(invocation, _error_from_exception(exc))
            raise
        handler.stop_invoke_agent(invocation)

    return _generator()


async def _direct_model_create_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
    if _suppress_direct_model_span.get() or not is_llm_span_enabled(
        default=True
    ):
        return await wrapped(*args, **kwargs)

    handler = _get_handler()
    invocation = _make_direct_model_invocation(instance, args, kwargs)
    handler.start_llm(invocation)
    try:
        result = await wrapped(*args, **kwargs)
        apply_create_result(invocation, result)
        handler.stop_llm(invocation)
        return result
    except Exception as exc:
        handler.fail_llm(invocation, _error_from_exception(exc))
        raise


def _direct_model_create_stream_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
    if _suppress_direct_model_span.get() or not is_llm_span_enabled(
        default=True
    ):
        return wrapped(*args, **kwargs)

    async def _generator():  # type: ignore[no-untyped-def]
        handler = _get_handler()
        invocation = _make_direct_model_invocation(instance, args, kwargs)
        handler.start_llm(invocation)
        try:
            async for item in wrapped(*args, **kwargs):
                if (
                    isinstance(item, str)
                    and invocation.monotonic_first_token_s is None
                ):
                    invocation.monotonic_first_token_s = timeit.default_timer()
                elif type(item).__name__ == "CreateResult":
                    apply_create_result(invocation, item)
                yield item
        except (GeneratorExit, asyncio.CancelledError) as exc:
            handler.fail_llm(invocation, _error_from_exception(exc))
            raise
        except Exception as exc:
            handler.fail_llm(invocation, _error_from_exception(exc))
            raise
        handler.stop_llm(invocation)

    return _generator()


async def _selector_select_speaker_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
    agent_name = _text_like(
        getattr(instance, "_name", None) or type(instance).__name__
    )
    token = _direct_model_agent_name.set(agent_name)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _direct_model_agent_name.reset(token)


def _text_like(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _call_llm_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
    if not is_llm_span_enabled(default=True):
        return wrapped(*args, **kwargs)

    async def _generator():  # type: ignore[no-untyped-def]
        handler = _get_handler()
        model_client = _arg_value(args, kwargs, "model_client")
        agent_name = _arg_value(args, kwargs, "agent_name")
        output_content_type = _arg_value(args, kwargs, "output_content_type")
        output_type = "json" if output_content_type is not None else None
        invocation = make_llm_invocation(
            model_client,
            await _collect_llm_messages(args, kwargs),
            await _collect_tools(args, kwargs),
            agent_name=agent_name,
            output_type=output_type,
        )
        handler.start_llm(invocation)
        try:
            async for item in _contextual_async_iter(
                wrapped(*args, **kwargs),
                _suppress_direct_model_span,
                True,
            ):
                if (
                    type(item).__name__ == "ModelClientStreamingChunkEvent"
                    and invocation.monotonic_first_token_s is None
                ):
                    invocation.monotonic_first_token_s = timeit.default_timer()
                elif type(item).__name__ == "CreateResult":
                    apply_create_result(invocation, item)
                yield item
        except (GeneratorExit, asyncio.CancelledError) as exc:
            handler.fail_llm(invocation, _error_from_exception(exc))
            raise
        except Exception as exc:
            handler.fail_llm(invocation, _error_from_exception(exc))
            raise
        handler.stop_llm(invocation)

    return _generator()


async def _execute_tool_call_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
    handler = _get_handler()
    tool_call = _named_arg_value(
        args, kwargs, "tool_call", _EXECUTE_TOOL_CALL_PARAM_NAMES
    )
    invocation = make_tool_invocation(tool_call)
    handler.start_execute_tool(invocation)
    token = _suppress_native_tool_span.set(True)
    try:
        result = await wrapped(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 2:
            apply_tool_result(invocation, result[1])
        handler.stop_execute_tool(invocation)
        return result
    except Exception as exc:
        handler.fail_execute_tool(invocation, _error_from_exception(exc))
        raise
    finally:
        _suppress_native_tool_span.reset(token)


@contextmanager
def _suppressed_native_tool_span():  # type: ignore[no-untyped-def]
    yield trace.get_current_span()


def _trace_tool_span_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
    if _suppress_native_tool_span.get():
        return _suppressed_native_tool_span()
    return wrapped(*args, **kwargs)


def _get_handler() -> ExtendedTelemetryHandler:
    if _get_handler.handler is None:
        _get_handler.handler = ExtendedTelemetryHandler()  # type: ignore[attr-defined]
    return _get_handler.handler


_get_handler.handler = None  # type: ignore[attr-defined]


def _wrap_optional(module: str, target: str, wrapper: Any) -> None:
    try:
        wrap_function_wrapper(module, target, wrapper)
    except (ImportError, AttributeError) as exc:
        logger.debug(
            "AutoGen optional patch skipped for %s.%s: %s",
            module,
            target,
            exc,
        )
        return
    _optional_wrappers.append((module, target))


def _unwrap_optional() -> None:
    while _optional_wrappers:
        module, target = _optional_wrappers.pop()
        try:
            unwrap(module, target)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "AutoGen optional unwrap failed for %s.%s: %s",
                module,
                target,
                exc,
            )


def apply_agentchat_patch(handler: ExtendedTelemetryHandler) -> None:
    global _applied
    if _applied:
        return
    _get_handler.handler = handler  # type: ignore[attr-defined]
    try:
        wrap_function_wrapper(
            _ASSISTANT_AGENT_MODULE,
            "AssistantAgent.on_messages_stream",
            _on_messages_stream_wrapper,
        )
        wrap_function_wrapper(
            _ASSISTANT_AGENT_MODULE,
            "AssistantAgent._call_llm",
            _call_llm_wrapper,
        )
        wrap_function_wrapper(
            _ASSISTANT_AGENT_MODULE,
            "AssistantAgent._execute_tool_call",
            _execute_tool_call_wrapper,
        )
        wrap_function_wrapper(
            _BASE_TOOL_MODULE,
            "trace_tool_span",
            _trace_tool_span_wrapper,
        )
    except (ImportError, AttributeError) as exc:
        for name in (
            "AssistantAgent.on_messages_stream",
            "AssistantAgent._call_llm",
            "AssistantAgent._execute_tool_call",
        ):
            try:
                unwrap(_ASSISTANT_AGENT_MODULE, name)
            except Exception:
                pass
        try:
            unwrap(_BASE_TOOL_MODULE, "trace_tool_span")
        except Exception:
            pass
        logger.warning("AutoGen AgentChat patch skipped: %s", exc)
        return

    for module, target in (
        (
            _CODE_EXECUTOR_AGENT_MODULE,
            "CodeExecutorAgent.on_messages_stream",
        ),
        (
            _SOCIETY_OF_MIND_AGENT_MODULE,
            "SocietyOfMindAgent.on_messages_stream",
        ),
        (_USER_PROXY_AGENT_MODULE, "UserProxyAgent.on_messages_stream"),
    ):
        _wrap_optional(module, target, _on_messages_stream_wrapper)

    _wrap_optional(
        _SELECTOR_GROUP_CHAT_MODULE,
        "SelectorGroupChatManager._select_speaker",
        _selector_select_speaker_wrapper,
    )
    _wrap_optional(
        _BASE_GROUP_CHAT_MODULE,
        "BaseGroupChat.run_stream",
        _team_run_stream_wrapper,
    )

    for module, cls_name in _DIRECT_MODEL_CLIENT_PATCHES:
        _wrap_optional(
            module, f"{cls_name}.create", _direct_model_create_wrapper
        )
        _wrap_optional(
            module,
            f"{cls_name}.create_stream",
            _direct_model_create_stream_wrapper,
        )

    _applied = True


def revert_agentchat_patch() -> None:
    global _applied
    if not _applied:
        return
    _unwrap_optional()
    try:
        unwrap(_ASSISTANT_AGENT_MODULE, "AssistantAgent.on_messages_stream")
        unwrap(_ASSISTANT_AGENT_MODULE, "AssistantAgent._call_llm")
        unwrap(_ASSISTANT_AGENT_MODULE, "AssistantAgent._execute_tool_call")
        unwrap(_BASE_TOOL_MODULE, "trace_tool_span")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("AutoGen AgentChat unwrap failed: %s", exc)
    _applied = False
