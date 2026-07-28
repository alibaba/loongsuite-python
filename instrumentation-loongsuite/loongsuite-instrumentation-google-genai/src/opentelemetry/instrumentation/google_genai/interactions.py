# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

try:
    # Google GenAI < 2.9.0
    from google.genai._interactions._streaming import Stream
    from google.genai._interactions.resources.interactions import (
        AsyncInteractionsResource,
        InteractionsResource,
    )
    from google.genai._interactions.types.interaction import Interaction, Usage
    from google.genai._interactions.types.interaction_create_params import (
        Input,
    )
    from google.genai._interactions.types.interaction_sse_event import (
        InteractionSSEEvent,
    )

    _HAS_INTERACTIONS = True
except ImportError:
    try:
        # Google GenAI >= 2.9.0
        from google.genai._gaos.interactions import (
            AsyncInteractions as AsyncInteractionsResource,
        )
        from google.genai._gaos.interactions import (
            Interactions as InteractionsResource,
        )
        from google.genai._gaos.interactions import (
            Stream,
        )
        from google.genai._gaos.types.interactions import (
            Interaction,
            InteractionSSEEvent,
            Usage,
        )
        from google.genai._gaos.types.interactions import (
            InteractionsInput as Input,
        )

        _HAS_INTERACTIONS = True
    except ImportError:
        _HAS_INTERACTIONS = False

        # Placeholders for older versions where interactions are not supported
        class InteractionsResource:
            create = None

        class AsyncInteractionsResource:
            create = None

        class Interaction:
            model = None
            usage = None

        class Usage:
            total_input_tokens = None
            total_output_tokens = None
            total_thought_tokens = None

        class Input:
            pass

        class InteractionSSEEvent:
            pass

        class Stream:
            pass


from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.google_genai.client_info import (
    get_client_info as _get_client_info,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai import hook_advice
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    Text,
    ToolCallResponse,
    Uri,
)
from opentelemetry.util.genai.types import (
    ToolCall as ToolCallRequest,
)

from ._compat import (
    GenericPart,
    InferenceInvocation,
    TelemetryHandler,
)
from ._stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)


class _InteractionsMethodsSnapshot:
    def __init__(self) -> None:
        self._original_create = InteractionsResource.create
        self._original_create_code = InteractionsResource.create.__code__
        self._original_async_create = AsyncInteractionsResource.create
        self._original_async_create_code = (
            AsyncInteractionsResource.create.__code__
        )

    def restore(self) -> None:
        self._original_create.__code__ = self._original_create_code
        self._original_async_create.__code__ = self._original_async_create_code

        InteractionsResource.create = self._original_create
        AsyncInteractionsResource.create = self._original_async_create


# Magic incantation used by native Google ADK instrumentation to identify
# instrumented functions and suppress its own internal tracing when OTel is active.
def _set_co_filename(wrapped: object) -> None:
    wrapped.__wrapped__.__code__ = wrapped.__wrapped__.__code__.replace(
        co_filename=__file__.replace("\\", "/")
    )


def _apply_interaction_response_attributes(
    response: Interaction,
    invocation: InferenceInvocation,
    telemetry_handler: TelemetryHandler,
) -> None:
    invocation.response_model_name = response.model

    usage = response.usage or Usage()

    invocation.input_tokens = usage.total_input_tokens
    invocation.output_tokens = usage.total_output_tokens
    invocation.thinking_tokens = usage.total_thought_tokens
    invocation.cache_read_input_tokens = usage.total_cached_tokens

    if telemetry_handler.should_capture_content():
        invocation.output_messages = _interactions_response_to_messages(
            response
        )


def _get_field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# Logic for parsing Input is tricky:
# https://github.com/open-telemetry/donation-openinference/blob/6cdd644d79fccf50aedcb614187f924ddfcafb7b/python/instrumentation/openinference-instrumentation-google-genai/src/openinference/instrumentation/google_genai/interactions_attributes.py#L103
# It doesn't make sense for this to be a List[InputMessage] (per semconv),
# because this API doesn't take conversation history as input (unlike the generate_content API).
# Conversation history is stored server-side and referenced via a interaction ID parameter.
def _interactions_input_to_messages(
    input_data: Input | None,
) -> list[InputMessage]:
    # None will end up raising an exception by the SDK
    if input_data is None:
        return []
    if isinstance(input_data, str):
        return [InputMessage(role="user", parts=[Text(content=input_data)])]

    if not isinstance(input_data, Sequence):
        input_data = [input_data]

    parts = []
    for item in input_data:
        item_type = _get_field(item, "type")
        if item_type == "function_call":
            call_id = _get_field(item, "id")
            name = _get_field(item, "name")
            arguments = _get_field(item, "arguments")
            part = ToolCallRequest(
                id=call_id, name=name or "", arguments=arguments
            )
            parts.append(part)
        elif item_type == "function_result":
            call_id = _get_field(item, "call_id")
            result = _get_field(item, "result")
            part = ToolCallResponse(id=call_id, response=result)
            parts.append(part)
        elif isinstance(item, str):
            parts.append(Text(content=item))
        elif item_type == "text":
            part = Text(content=_get_field(item, "text") or "")
            parts.append(part)
        elif item_type == "document":
            part = Uri(
                mime_type=_get_field(item, "mime_type"),
                modality="document",
                uri=_get_field(item, "uri") or "",
            )
            parts.append(part)
        elif item_type is not None:
            part = GenericPart(value=type(item).__name__)
            parts.append(part)

    return [InputMessage(role="user", parts=parts)]


def _get_interaction_output_text(interaction: Interaction) -> str:
    if getattr(interaction, "output_text", None):
        return interaction.output_text

    texts = []
    if interaction.steps:
        for step in interaction.steps:
            if getattr(step, "type", None) == "model_output":
                content = getattr(step, "content", None)
                if content:
                    for item in content:
                        if getattr(item, "type", None) == "text" and hasattr(
                            item, "text"
                        ):
                            texts.append(item.text)
    return "".join(texts)


# It doesn't make sense for this to be a list of OutputMessage (per semconv),
# because this API doesn't return conversation history as output (unlike the generate_content API).
# Model's response is returned as a list of steps:
# https://ai.google.dev/gemini-api/docs/migrate-to-interactions#basic-input-output
# https://ai.google.dev/api/interactions-api#Resource:Step
def _interactions_response_to_messages(
    interaction: Interaction,
) -> list[OutputMessage]:
    output_text = _get_interaction_output_text(interaction)
    return [
        OutputMessage(
            role="assistant",
            parts=[Text(content=output_text)],
            finish_reason="stop",
        )
    ]


class InteractionsStreamWrapper(SyncStreamWrapper[InteractionSSEEvent]):
    def __init__(
        self,
        stream: Iterable[InteractionSSEEvent],
        invocation: InferenceInvocation,
        telemetry_handler: TelemetryHandler,
    ) -> None:
        super().__init__(stream)
        self._self_invocation = invocation
        self._self_telemetry_handler = telemetry_handler
        self._self_last_interaction: Interaction | None = None

    def _process_chunk(self, chunk: InteractionSSEEvent) -> None:
        self._self_invocation.record_first_token()
        event_type = _get_field(chunk, "event_type")
        if event_type in ("interaction_completed", "interaction.completed"):
            interaction = _get_field(chunk, "interaction")
            if interaction:
                self._self_last_interaction = interaction

    def _on_stream_end(self) -> None:
        if self._self_last_interaction:
            _apply_interaction_response_attributes(
                self._self_last_interaction,
                self._self_invocation,
                self._self_telemetry_handler,
            )
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


class AsyncInteractionsStreamWrapper(AsyncStreamWrapper[InteractionSSEEvent]):
    def __init__(
        self,
        stream: AsyncIterable[InteractionSSEEvent],
        invocation: InferenceInvocation,
        telemetry_handler: TelemetryHandler,
    ) -> None:
        super().__init__(stream)
        self._self_invocation = invocation
        self._self_telemetry_handler = telemetry_handler
        self._self_last_interaction: Interaction | None = None

    def _process_chunk(self, chunk: InteractionSSEEvent) -> None:
        self._self_invocation.record_first_token()
        event_type = _get_field(chunk, "event_type")
        if event_type in ("interaction_completed", "interaction.completed"):
            interaction = _get_field(chunk, "interaction")
            if interaction:
                self._self_last_interaction = interaction

    def _on_stream_end(self) -> None:
        if self._self_last_interaction:
            _apply_interaction_response_attributes(
                self._self_last_interaction,
                self._self_invocation,
                self._self_telemetry_handler,
            )
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


@dataclass
class _InteractionsAdviceState:
    invocation: InferenceInvocation
    is_stream: bool


@hook_advice("google-genai", "prepare_interactions")
def _prepare_interactions_advice(
    telemetry_handler: TelemetryHandler,
    instance: Any,
    kwargs: dict[str, Any],
) -> _InteractionsAdviceState:
    invocation = None
    try:
        is_vertex, server_address = _get_client_info(instance)
        invocation = telemetry_handler.inference(
            provider=(
                GenAIAttributes.GenAiSystemValues.VERTEX_AI.value
                if is_vertex
                else GenAIAttributes.GenAiSystemValues.GEMINI.value
            ),
            request_model=kwargs.get("model") or kwargs.get("agent"),
            operation_name="interactions.create",
            server_address=server_address,
        )
        if telemetry_handler.should_capture_content():
            invocation.input_messages = _interactions_input_to_messages(
                kwargs.get("input")
            )
            if system_instruction := kwargs.get("system_instruction"):
                invocation.system_instruction = [
                    Text(content=system_instruction)
                ]
        return _InteractionsAdviceState(
            invocation=invocation,
            is_stream=bool(kwargs.get("stream", False)),
        )
    except Exception:
        if invocation is not None:
            telemetry_handler.abandon_inference(invocation)
        raise


@hook_advice("google-genai", "complete_interactions")
def _complete_interactions_advice(
    telemetry_handler: TelemetryHandler,
    state: _InteractionsAdviceState,
    response: Interaction,
) -> None:
    try:
        _apply_interaction_response_attributes(
            response,
            state.invocation,
            telemetry_handler,
        )
    finally:
        state.invocation.stop()


@hook_advice("google-genai", "fail_interactions")
def _fail_interactions_advice(
    state: _InteractionsAdviceState,
    error: BaseException,
) -> None:
    state.invocation.fail(error)


@hook_advice("google-genai", "detach_interactions_stream_context")
def _detach_interactions_stream_context_advice(
    telemetry_handler: TelemetryHandler,
    state: _InteractionsAdviceState,
) -> bool:
    telemetry_handler.detach_inference_context(state.invocation)
    return True


@hook_advice("google-genai", "abandon_interactions")
def _abandon_interactions_advice(
    telemetry_handler: TelemetryHandler,
    state: _InteractionsAdviceState,
) -> None:
    telemetry_handler.abandon_inference(state.invocation)


@hook_advice("google-genai", "wrap_interactions_stream")
def _wrap_interactions_stream_advice(
    stream: Any,
    state: _InteractionsAdviceState,
    telemetry_handler: TelemetryHandler,
    *,
    asynchronous: bool,
) -> Any:
    wrapper_type = (
        AsyncInteractionsStreamWrapper
        if asynchronous
        else InteractionsStreamWrapper
    )
    return wrapper_type(
        stream,
        state.invocation,
        telemetry_handler,
    )


def _create_instrumented_interactions_create(
    telemetry_handler: TelemetryHandler,
) -> Callable[
    [
        Callable[..., Interaction | Stream[InteractionSSEEvent]],
        InteractionsResource,
        tuple[Any, ...],
        dict[str, Any],
    ],
    Interaction | InteractionsStreamWrapper,
]:
    def instrumented_interactions_create(
        wrapped: Callable[..., Interaction | Stream[InteractionSSEEvent]],
        instance: InteractionsResource,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Interaction | InteractionsStreamWrapper:
        state = _prepare_interactions_advice(
            telemetry_handler,
            instance,
            kwargs,
        )
        try:
            response = wrapped(*args, **kwargs)
        except BaseException as error:
            if state is not None:
                _fail_interactions_advice(state, error)
            raise
        if state is None:
            return response
        if not state.is_stream:
            _complete_interactions_advice(
                telemetry_handler,
                state,
                cast(Interaction, response),
            )
            return response
        if not _detach_interactions_stream_context_advice(
            telemetry_handler,
            state,
        ):
            _abandon_interactions_advice(telemetry_handler, state)
            return response
        stream_wrapper = _wrap_interactions_stream_advice(
            response,
            state,
            telemetry_handler,
            asynchronous=False,
        )
        if stream_wrapper is None:
            _abandon_interactions_advice(telemetry_handler, state)
            return response
        return stream_wrapper

    return instrumented_interactions_create


def _create_instrumented_async_interactions_create(
    telemetry_handler: TelemetryHandler,
) -> Callable[
    [
        Callable[..., Any],
        AsyncInteractionsResource,
        tuple[Any, ...],
        dict[str, Any],
    ],
    Any,
]:
    async def instrumented_interactions_create(
        wrapped: Callable[..., Any],
        instance: AsyncInteractionsResource,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Interaction | AsyncInteractionsStreamWrapper:
        state = _prepare_interactions_advice(
            telemetry_handler,
            instance,
            kwargs,
        )
        try:
            response = await wrapped(*args, **kwargs)
        except BaseException as error:
            if state is not None:
                _fail_interactions_advice(state, error)
            raise
        if state is None:
            return response
        if not state.is_stream:
            _complete_interactions_advice(
                telemetry_handler,
                state,
                cast(Interaction, response),
            )
            return response
        if not _detach_interactions_stream_context_advice(
            telemetry_handler,
            state,
        ):
            _abandon_interactions_advice(telemetry_handler, state)
            return response
        stream_wrapper = _wrap_interactions_stream_advice(
            response,
            state,
            telemetry_handler,
            asynchronous=True,
        )
        if stream_wrapper is None:
            _abandon_interactions_advice(telemetry_handler, state)
            return response
        return stream_wrapper

    return instrumented_interactions_create


def uninstrument_interactions(snapshot: object) -> None:
    if snapshot is None:
        return
    assert isinstance(snapshot, _InteractionsMethodsSnapshot)
    snapshot.restore()


def instrument_interactions(
    telemetry_handler: TelemetryHandler,
) -> object | None:
    if not _HAS_INTERACTIONS:
        return None

    snapshot = _InteractionsMethodsSnapshot()

    try:
        import google.genai._interactions.resources.interactions  # noqa: F401, PLC0415

        module_path = "google.genai._interactions.resources.interactions"
        sync_class = "InteractionsResource"
        async_class = "AsyncInteractionsResource"
    except ImportError:
        # In version 2.9 of google-genai these were moved.
        module_path = "google.genai._gaos.interactions"
        sync_class = "Interactions"
        async_class = "AsyncInteractions"

    wrapped = wrap_function_wrapper(
        module_path,
        f"{sync_class}.create",
        _create_instrumented_interactions_create(telemetry_handler),
    )
    _set_co_filename(wrapped)
    wrapped2 = wrap_function_wrapper(
        module_path,
        f"{async_class}.create",
        _create_instrumented_async_interactions_create(telemetry_handler),
    )
    _set_co_filename(wrapped2)
    return snapshot
