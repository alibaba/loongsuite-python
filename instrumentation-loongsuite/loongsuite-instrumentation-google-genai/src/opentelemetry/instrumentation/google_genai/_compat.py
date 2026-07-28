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

"""Bridge the upstream GenAI 1.0 invocation API to LoongSuite extensions.

The canonical Google GenAI instrumentation uses the invocation-oriented API
introduced by ``opentelemetry-util-genai`` 1.0. LoongSuite still exposes the
older lifecycle API plus ``ExtendedTelemetryHandler``. Keeping that impedance
match in this module lets the provider instrumentation stay close to upstream
and makes this bridge removable after the shared util migrates to 1.x.
"""

from __future__ import annotations

import logging
import os
import timeit
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any

from opentelemetry import context as context_api
from opentelemetry._logs import LogRecord, get_logger
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import set_span_in_context
from opentelemetry.util.genai.completion_hook import (
    CompletionHook,
    load_completion_hook,
)
from opentelemetry.util.genai.extended_handler import (
    ExtendedTelemetryHandler,
)
from opentelemetry.util.genai.extended_types import (
    EmbeddingInvocation as LegacyEmbeddingInvocation,
)
from opentelemetry.util.genai.extended_types import (
    ExecuteToolInvocation as LegacyToolInvocation,
)
from opentelemetry.util.genai.span_utils import (
    _get_llm_common_attributes,
    _get_llm_request_attributes,
    _get_llm_response_attributes,
)
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    Error,
    LLMInvocation,
)
from opentelemetry.util.genai.utils import (
    gen_ai_json_dumps,
    should_emit_event,
)

_logger = logging.getLogger(__name__)
_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"
_FINISH_REASONS = "gen_ai.response.finish_reasons"
_INPUT_MESSAGES = "gen_ai.input.messages"
_OUTPUT_MESSAGES = "gen_ai.output.messages"
_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
_TOOL_DEFINITIONS = "gen_ai.tool.definitions"


def _is_suppressed() -> bool:
    # OSS honors only the standard OpenTelemetry suppression contract. Robin
    # extends this provider-level check with its private
    # _SUPPRESS_LLM_SDK_KEY after synchronization. Keep that check here rather
    # than in ExtendedTelemetryHandler.start_llm(): high-level framework
    # instrumentations use the same handler and must retain their own spans.
    return bool(context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY))


def _capture_mode() -> ContentCapturingMode:
    configured = os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
    )
    if not configured:
        return ContentCapturingMode.NO_CONTENT
    try:
        return ContentCapturingMode[configured.upper()]
    except KeyError:
        return ContentCapturingMode.NO_CONTENT


def _captures_content() -> bool:
    return _capture_mode() is not ContentCapturingMode.NO_CONTENT


def _captures_content_on_span() -> bool:
    return _capture_mode() in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    )


@dataclass
class GenericPart:
    """Provider-specific message part used by the upstream interactions parser."""

    value: Any
    type: str = "generic"


class _InvocationLifecycle:
    _finished: bool

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_value is not None:
            self.fail(exc_value)
        else:
            self.stop()


class InferenceInvocation(LLMInvocation, _InvocationLifecycle):
    """Upstream-style inference invocation backed by the extended handler."""

    def __init__(
        self,
        owner: TelemetryHandler,
        provider: str,
        *,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        operation_name: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__(
            request_model=request_model,
            provider=provider,
            operation_name=operation_name or "chat",
            server_address=server_address,
            server_port=server_port,
        )
        self._owner = owner
        self._finished = False
        self.thinking_tokens: int | None = None
        if not _is_suppressed():
            owner._handler.start_llm(self)

    @property
    def request_choice_count(self) -> int | None:
        return self.choice_count

    @request_choice_count.setter
    def request_choice_count(self, value: int | None) -> None:
        self.choice_count = value

    @property
    def cache_creation_input_tokens(self) -> int | None:
        return self.usage_cache_creation_input_tokens

    @cache_creation_input_tokens.setter
    def cache_creation_input_tokens(self, value: int | None) -> None:
        self.usage_cache_creation_input_tokens = value

    @property
    def cache_read_input_tokens(self) -> int | None:
        return self.usage_cache_read_input_tokens

    @cache_read_input_tokens.setter
    def cache_read_input_tokens(self, value: int | None) -> None:
        self.usage_cache_read_input_tokens = value

    def record_first_token(self) -> None:
        if self.monotonic_first_token_s is None:
            self.monotonic_first_token_s = timeit.default_timer()

    def _finish_reasons(self) -> list[str]:
        if self.finish_reasons is not None:
            return self.finish_reasons
        return [
            message.finish_reason
            for message in self.output_messages
            if message.finish_reason
        ]

    def _prepare_finish(
        self, error: Error | BaseException | None = None
    ) -> None:
        custom_attributes = dict(self.attributes)
        if self.thinking_tokens is not None:
            self.attributes[_REASONING_OUTPUT_TOKENS] = self.thinking_tokens
            self.output_tokens = (
                self.output_tokens or 0
            ) + self.thinking_tokens
        if finish_reasons := self._finish_reasons():
            self.attributes[_FINISH_REASONS] = finish_reasons
        if self.tool_definitions:
            self.attributes[_TOOL_DEFINITIONS] = gen_ai_json_dumps(
                [asdict(tool) for tool in self.tool_definitions]
            )
        if _captures_content_on_span():
            message_attributes = (
                (_INPUT_MESSAGES, self.input_messages),
                (_OUTPUT_MESSAGES, self.output_messages),
                (_SYSTEM_INSTRUCTIONS, self.system_instruction),
            )
            for key, messages in message_attributes:
                if messages:
                    self.attributes[key] = gen_ai_json_dumps(
                        [asdict(message) for message in messages]
                    )
        log_record = self._owner._create_event(
            self,
            custom_attributes=custom_attributes,
            error=error,
        )
        self._owner._notify_completion(self, log_record=log_record)
        if log_record is not None:
            self._owner._logger.emit(log_record)

    def stop(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self.span is None:
            return
        self._prepare_finish()
        self._owner._handler.stop_llm(self)

    def fail(self, error: Error | BaseException) -> None:
        if self._finished:
            return
        self._finished = True
        if self.span is None:
            return
        if not isinstance(error, Error):
            error = Error(message=str(error), type=type(error))
        self._prepare_finish(error)
        self._owner._handler.fail_llm(self, error)


class EmbeddingInvocation(LegacyEmbeddingInvocation, _InvocationLifecycle):
    """Upstream-style embedding invocation backed by the extended handler."""

    def __init__(
        self,
        owner: TelemetryHandler,
        provider: str,
        *,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
    ) -> None:
        super().__init__(
            request_model=request_model or "",
            provider=provider,
            server_address=server_address,
            server_port=server_port,
        )
        self._owner = owner
        self._finished = False
        self.metric_attributes: dict[str, Any] = {}
        if not _is_suppressed():
            owner._handler.start_embedding(self)

    def stop(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self.span is not None:
            self._owner._handler.stop_embedding(self)

    def fail(self, error: Error | BaseException) -> None:
        if self._finished:
            return
        self._finished = True
        if self.span is None:
            return
        if not isinstance(error, Error):
            error = Error(message=str(error), type=type(error))
        self._owner._handler.fail_embedding(self, error)


class ToolInvocation(LegacyToolInvocation, _InvocationLifecycle):
    """Upstream-style tool invocation backed by the extended handler."""

    def __init__(
        self,
        owner: TelemetryHandler,
        name: str,
        *,
        tool_call_id: str | None = None,
        tool_type: str | None = None,
        tool_description: str | None = None,
    ) -> None:
        super().__init__(
            tool_name=name,
            tool_call_id=tool_call_id,
            tool_type=tool_type,
            tool_description=tool_description,
        )
        self._owner = owner
        self._finished = False
        self.should_capture_content_on_span = _captures_content_on_span()
        self.metric_attributes: dict[str, Any] = {}
        if not _is_suppressed():
            owner._handler.start_execute_tool(self)

    @property
    def arguments(self) -> Any:
        return self.tool_call_arguments

    @arguments.setter
    def arguments(self, value: Any) -> None:
        self.tool_call_arguments = value

    @property
    def tool_result(self) -> Any:
        return self.tool_call_result

    @tool_result.setter
    def tool_result(self, value: Any) -> None:
        self.tool_call_result = value

    def stop(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self.span is not None:
            self._owner._handler.stop_execute_tool(self)

    def fail(self, error: Error | BaseException) -> None:
        if self._finished:
            return
        self._finished = True
        if self.span is None:
            return
        if not isinstance(error, Error):
            error = Error(message=str(error), type=type(error))
        self._owner._handler.fail_execute_tool(self, error)


class TelemetryHandler:
    """Small 1.0-compatible facade over ``ExtendedTelemetryHandler``."""

    def __init__(
        self,
        *,
        tracer_provider=None,
        meter_provider=None,
        logger_provider=None,
        completion_hook: CompletionHook | None = None,
    ) -> None:
        self._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        # LoongSuite 0.x util emits a legacy event shape. This facade owns the
        # upstream 1.0 event so only one operation-details event is produced.
        self._handler._logger = None  # pylint: disable=protected-access
        self._logger = get_logger(
            __name__,
            logger_provider=logger_provider,
        )
        self._completion_hook = completion_hook or load_completion_hook()

    def inference(
        self,
        provider: str,
        *,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        operation_name: str | None = None,
        **kwargs: Any,
    ) -> InferenceInvocation:
        return InferenceInvocation(
            self,
            provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
            operation_name=operation_name,
            **kwargs,
        )

    def embedding(
        self,
        provider: str,
        *,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
    ) -> EmbeddingInvocation:
        return EmbeddingInvocation(
            self,
            provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
        )

    def tool(
        self,
        name: str,
        *,
        tool_call_id: str | None = None,
        tool_type: str | None = None,
        tool_description: str | None = None,
    ) -> ToolInvocation:
        return ToolInvocation(
            self,
            name,
            tool_call_id=tool_call_id,
            tool_type=tool_type,
            tool_description=tool_description,
        )

    def should_capture_content(self) -> bool:
        return _captures_content()

    def _create_event(
        self,
        invocation: InferenceInvocation,
        *,
        custom_attributes: dict[str, Any],
        error: Error | BaseException | None,
    ) -> LogRecord | None:
        if not should_emit_event():
            return None
        attributes: dict[str, Any] = {}
        attributes.update(_get_llm_common_attributes(invocation))
        attributes.update(_get_llm_request_attributes(invocation))
        attributes.update(_get_llm_response_attributes(invocation))
        attributes.update(custom_attributes)
        if invocation.thinking_tokens is not None:
            attributes[_REASONING_OUTPUT_TOKENS] = invocation.thinking_tokens
        if finish_reasons := invocation._finish_reasons():
            attributes[_FINISH_REASONS] = finish_reasons
        if invocation.tool_definitions:
            attributes[_TOOL_DEFINITIONS] = [
                asdict(tool) for tool in invocation.tool_definitions
            ]
        if _capture_mode() in (
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ):
            message_attributes = (
                (_INPUT_MESSAGES, invocation.input_messages),
                (_OUTPUT_MESSAGES, invocation.output_messages),
                (_SYSTEM_INSTRUCTIONS, invocation.system_instruction),
            )
            for key, messages in message_attributes:
                if messages:
                    attributes[key] = [asdict(message) for message in messages]
        if error is not None:
            error_type = (
                error.type if isinstance(error, Error) else type(error)
            )
            attributes["error.type"] = error_type.__qualname__
        return LogRecord(
            event_name="gen_ai.client.inference.operation.details",
            attributes=attributes,
            context=set_span_in_context(invocation.span),
        )

    def _notify_completion(
        self,
        invocation: InferenceInvocation,
        *,
        log_record: LogRecord | None,
    ) -> None:
        if invocation.span is None:
            return
        try:
            self._completion_hook.on_completion(
                inputs=invocation.input_messages,
                outputs=invocation.output_messages,
                system_instruction=invocation.system_instruction,
                tool_definitions=invocation.tool_definitions,
                span=invocation.span,
                log_record=log_record,
            )
        except Exception:
            _logger.debug(
                "Google GenAI completion hook failed",
                exc_info=True,
            )


__all__ = [
    "EmbeddingInvocation",
    "GenericPart",
    "InferenceInvocation",
    "TelemetryHandler",
    "ToolInvocation",
]
