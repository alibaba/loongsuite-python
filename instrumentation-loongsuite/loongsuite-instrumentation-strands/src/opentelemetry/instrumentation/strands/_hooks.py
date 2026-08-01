import logging
from typing import Any

from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
)
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
)

logger = logging.getLogger(__name__)


class LoongsuiteHook:
    """Hook that creates loongsuite-conformant spans for Strands Agent execution.

    Leverages the Strands hooks system to observe agent invocations, model calls,
    and tool executions, producing spans that follow the OTel GenAI semantic
    conventions as extended by loongsuite.

    Correlates before/after events using invocation_state identity (shared object
    passed to both before and after events in the Strands hooks system).
    """

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler
        self._agent_invocations: dict[int, InvokeAgentInvocation] = {}
        self._model_invocations: dict[int, LLMInvocation] = {}
        self._tool_invocations: list[ExecuteToolInvocation] = []
        self._react_invocations: dict[int, ReactStepInvocation] = {}
        self._cycle_counters: dict[int, int] = {}

    def __call__(self, event: Any) -> None:
        event_type = type(event).__name__
        handler_method = getattr(self, f"_on_{_camel_to_snake(event_type)}", None)
        if handler_method:
            try:
                handler_method(event)
            except Exception:
                logger.debug("Error handling strands event %s", event_type, exc_info=True)

    def _on_before_invocation_event(self, event: Any) -> None:
        agent = getattr(event, "agent", None)
        agent_name = _get_agent_name(agent)
        agent_id = _get_agent_id(agent)
        model_name = _get_model_name(agent)

        input_messages = _extract_input_messages(event)

        invocation = InvokeAgentInvocation(
            provider="strands",
            agent_name=agent_name,
            agent_id=agent_id,
            request_model=model_name,
            input_messages=input_messages,
        )
        self._handler.start_invoke_agent(invocation)

        key = _invocation_key(event)
        self._agent_invocations[key] = invocation
        self._cycle_counters[key] = 0

    def _on_after_invocation_event(self, event: Any) -> None:
        key = _invocation_key(event)
        invocation = self._agent_invocations.pop(key, None)
        self._cycle_counters.pop(key, None)

        if invocation is None:
            return

        result = getattr(event, "result", None)
        if result is not None:
            invocation.output_messages = _extract_output_messages_from_result(result)
            invocation.finish_reasons = [_get_stop_reason(result)]

        self._handler.stop_invoke_agent(invocation)

    def _on_before_model_call_event(self, event: Any) -> None:
        agent = getattr(event, "agent", None)
        model_name = _get_model_name(agent)

        invocation = LLMInvocation(
            request_model=model_name,
            operation_name="chat",
            provider="strands",
        )

        invocation_state = getattr(event, "invocation_state", None)
        if invocation_state is not None:
            messages = getattr(invocation_state, "messages", None)
            if messages:
                invocation.input_messages = _convert_strands_messages(messages)

        self._handler.start_llm(invocation)

        key = _invocation_key(event)
        self._model_invocations[key] = invocation

        self._start_react_step(key)

    def _on_after_model_call_event(self, event: Any) -> None:
        key = _invocation_key(event)
        invocation = self._model_invocations.pop(key, None)

        if invocation is not None:
            stop_response = getattr(event, "stop_response", None)
            exception = getattr(event, "exception", None)

            if exception is not None:
                self._handler.fail_llm(
                    invocation, Error(message=str(exception), type=type(exception))
                )
            else:
                if stop_response is not None:
                    invocation.output_messages = _extract_output_from_stop_response(stop_response)
                    invocation.finish_reasons = [_get_stop_reason_from_response(stop_response)]
                    usage = getattr(stop_response, "usage", None)
                    if usage:
                        invocation.input_tokens = getattr(usage, "input_tokens", None)
                        invocation.output_tokens = getattr(usage, "output_tokens", None)
                self._handler.stop_llm(invocation)

        self._stop_react_step(key, event)

    def _on_before_tool_call_event(self, event: Any) -> None:
        tool_use = getattr(event, "tool_use", None)
        selected_tool = getattr(event, "selected_tool", None)

        tool_name = "unknown"
        tool_call_id = None
        tool_arguments = None

        if tool_use:
            tool_name = getattr(tool_use, "name", None) or tool_name
            tool_call_id = getattr(tool_use, "tool_use_id", None)
            tool_arguments = getattr(tool_use, "input", None)
        elif selected_tool:
            tool_name = getattr(selected_tool, "name", None) or tool_name

        invocation = ExecuteToolInvocation(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_call_arguments=tool_arguments,
            provider="strands",
        )
        self._handler.start_execute_tool(invocation)
        self._tool_invocations.append(invocation)

    def _on_after_tool_call_event(self, event: Any) -> None:
        if not self._tool_invocations:
            return
        invocation = self._tool_invocations.pop()

        result = getattr(event, "result", None)
        if result is not None:
            invocation.tool_call_result = _serialize_tool_result(result)

        self._handler.stop_execute_tool(invocation)

    def _start_react_step(self, key: int) -> None:
        count = self._cycle_counters.get(key, 0) + 1
        self._cycle_counters[key] = count

        invocation = ReactStepInvocation(round=count)
        self._handler.start_react_step(invocation)
        self._react_invocations[key] = invocation

    def _stop_react_step(self, key: int, event: Any) -> None:
        invocation = self._react_invocations.pop(key, None)

        if invocation is None:
            return

        stop_response = getattr(event, "stop_response", None)
        if stop_response:
            invocation.finish_reason = _get_stop_reason_from_response(stop_response)

        self._handler.stop_react_step(invocation)


def _invocation_key(event: Any) -> int:
    invocation_state = getattr(event, "invocation_state", None)
    if invocation_state is not None:
        return id(invocation_state)
    return id(event)


def _camel_to_snake(name: str) -> str:
    result = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0:
            result.append("_")
        result.append(c.lower())
    return "".join(result)


def _get_agent_name(agent: Any) -> str:
    if agent is None:
        return "strands_agent"
    name = getattr(agent, "name", None)
    if name:
        return str(name)
    return "strands_agent"


def _get_agent_id(agent: Any) -> str | None:
    if agent is None:
        return None
    return getattr(agent, "agent_id", None)


def _get_model_name(agent: Any) -> str | None:
    if agent is None:
        return None
    model = getattr(agent, "model", None)
    if model is None:
        return None
    model_id = getattr(model, "model_id", None)
    if model_id:
        return str(model_id)
    return getattr(model, "name", None)


def _extract_input_messages(event: Any) -> list[InputMessage]:
    invocation_state = getattr(event, "invocation_state", None)
    if invocation_state is not None:
        raw_messages = getattr(invocation_state, "messages", None)
        if raw_messages:
            return _convert_strands_messages(raw_messages)

    raw_messages = getattr(event, "messages", None)
    if raw_messages:
        return _convert_strands_messages(raw_messages)
    return []


def _convert_strands_messages(messages: list) -> list[InputMessage]:
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                result.append(InputMessage(role=role, parts=[Text(content=content)]))
            elif isinstance(content, list):
                parts = _convert_content_blocks(content)
                result.append(InputMessage(role=role, parts=parts))
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                result.append(InputMessage(role=role, parts=[Text(content=content)]))
            elif isinstance(content, list):
                parts = _convert_content_blocks(content)
                result.append(InputMessage(role=role, parts=parts))
    return result


def _convert_content_blocks(blocks: list) -> list:
    parts = []
    for block in blocks:
        if isinstance(block, dict):
            block_type = block.get("type", "text")
            if block_type == "text":
                parts.append(Text(content=block.get("text", "")))
            elif block_type == "tool_use":
                parts.append(
                    ToolCall(
                        name=block.get("name", ""),
                        arguments=block.get("input"),
                        id=block.get("id"),
                    )
                )
            elif block_type == "tool_result":
                parts.append(
                    ToolCallResponse(
                        response=block.get("content", ""),
                        id=block.get("tool_use_id"),
                    )
                )
        elif isinstance(block, str):
            parts.append(Text(content=block))
    return parts


def _extract_output_messages_from_result(result: Any) -> list[OutputMessage]:
    messages = []
    if isinstance(result, dict):
        content = result.get("content", "")
        if isinstance(content, str) and content:
            messages.append(
                OutputMessage(role="assistant", parts=[Text(content=content)], finish_reason="stop")
            )
    elif isinstance(result, str) and result:
        messages.append(
            OutputMessage(role="assistant", parts=[Text(content=result)], finish_reason="stop")
        )
    else:
        message = getattr(result, "message", None)
        if message:
            content = getattr(message, "content", None) or getattr(message, "text", None)
            if content and isinstance(content, str):
                messages.append(
                    OutputMessage(
                        role="assistant", parts=[Text(content=content)], finish_reason="stop"
                    )
                )
    return messages


def _extract_output_from_stop_response(stop_response: Any) -> list[OutputMessage]:
    messages = []
    message = getattr(stop_response, "message", None)
    if message:
        content = getattr(message, "content", None)
        if content and isinstance(content, str):
            messages.append(
                OutputMessage(role="assistant", parts=[Text(content=content)], finish_reason="stop")
            )
        elif content and isinstance(content, list):
            parts = _convert_content_blocks(content)
            messages.append(OutputMessage(role="assistant", parts=parts, finish_reason="stop"))
    return messages


def _get_stop_reason(result: Any) -> str:
    if isinstance(result, dict):
        return result.get("stop_reason", "stop") or "stop"
    stop_reason = getattr(result, "stop_reason", None)
    return str(stop_reason) if stop_reason else "stop"


def _get_stop_reason_from_response(stop_response: Any) -> str:
    stop_reason = getattr(stop_response, "stop_reason", None)
    if stop_reason:
        return str(stop_reason)
    return "stop"


def _serialize_tool_result(result: Any) -> Any:
    if isinstance(result, (str, int, float, bool, type(None))):
        return result
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        return [_serialize_tool_result(item) for item in result]
    content = getattr(result, "content", None)
    if content is not None:
        return _serialize_tool_result(content)
    return str(result)
