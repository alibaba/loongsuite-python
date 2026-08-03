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

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterable, Callable, Iterable
from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Union

from google.genai.models import AsyncModels, Models
from google.genai.models import t as transformers
from google.genai.types import (
    ContentListUnion,
    ContentListUnionDict,
    GenerateContentConfig,
    GenerateContentConfigOrDict,
    GenerateContentResponse,
    Tool,
    ToolUnionDict,
)
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes,
)
from opentelemetry.util.genai import async_hook_advice, hook_advice
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    GenericToolDefinition,
    ToolDefinition,
)
from opentelemetry.util.types import AttributeValue

from ._compat import InferenceInvocation, TelemetryHandler
from ._stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from .allowlist_util import AllowList
from .client_info import get_client_info as _get_client_info
from .custom_semconv import GCP_GENAI_OPERATION_CONFIG
from .dict_util import flatten_dict
from .message import (
    StreamOutputMessageAccumulator,
    to_input_messages,
    to_output_messages,
    to_system_instructions,
)
from .tool_call_wrapper import wrapped_tool

_is_mcp_imported = False
McpClientSession = McpTool = None
try:
    from mcp import ClientSession as McpClientSession
    from mcp import Tool as McpTool

    _is_mcp_imported = True
except ImportError:
    pass

GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY = context_api.create_key(
    "generate_content_extra_attributes_context_key"
)


class _MethodsSnapshot:
    def __init__(self):
        self._original_generate_content = Models.generate_content
        self._original_generate_content_code = Models.generate_content.__code__
        self._original_generate_content_stream = Models.generate_content_stream
        self._original_generate_content_stream_code = (
            Models.generate_content_stream.__code__
        )
        self._original_async_generate_content = AsyncModels.generate_content
        self._original_async_generate_content_code = (
            AsyncModels.generate_content.__code__
        )
        self._original_async_generate_content_stream = (
            AsyncModels.generate_content_stream
        )
        self._original_async_generate_content_stream_code = (
            AsyncModels.generate_content_stream.__code__
        )

    @property
    def generate_content(self):
        return self._original_generate_content

    @property
    def generate_content_stream(self):
        return self._original_generate_content_stream

    @property
    def async_generate_content(self):
        return self._original_async_generate_content

    @property
    def async_generate_content_stream(self):
        return self._original_async_generate_content_stream

    def restore(self):
        self._original_generate_content.__code__ = (
            self._original_generate_content_code
        )
        self._original_generate_content_stream.__code__ = (
            self._original_generate_content_stream_code
        )
        self._original_async_generate_content.__code__ = (
            self._original_async_generate_content_code
        )
        self._original_async_generate_content_stream.__code__ = (
            self._original_async_generate_content_stream_code
        )

        Models.generate_content = self._original_generate_content
        Models.generate_content_stream = self._original_generate_content_stream
        AsyncModels.generate_content = self._original_async_generate_content
        AsyncModels.generate_content_stream = (
            self._original_async_generate_content_stream
        )


# Magic incantation used by native Google ADK instrumentation to identify
# instrumented functions and suppress its own internal tracing when OTel is active.
def _set_co_filename(wrapped: object) -> None:
    wrapped.__wrapped__.__code__ = wrapped.__wrapped__.__code__.replace(
        co_filename=__file__.replace("\\", "/")
    )


def _guess_genai_system_from_env():
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "0").lower() in [
        "true",
        "1",
    ]:
        return gen_ai_attributes.GenAiSystemValues.VERTEX_AI.name.lower()
    return gen_ai_attributes.GenAiSystemValues.GEMINI.name.lower()


def _get_is_vertexai(models_object: Union[Models, AsyncModels]):
    # Since commit 8e561de04965bb8766db87ad8eea7c57c1040442 of "googleapis/python-genai",
    # it is possible to obtain the information using a documented property.
    if hasattr(models_object, "vertexai"):
        vertexai_attr = getattr(models_object, "vertexai")
        if vertexai_attr is not None:
            return vertexai_attr
    # For earlier revisions, it is necessary to deeply inspect the internals.
    if hasattr(models_object, "_api_client"):
        client = getattr(models_object, "_api_client")
        if not client:
            return None
        if hasattr(client, "vertexai"):
            return getattr(client, "vertexai")
    return None


def _determine_genai_system(models_object: Union[Models, AsyncModels]):
    vertexai_attr = _get_is_vertexai(models_object)
    if vertexai_attr is None:
        return _guess_genai_system_from_env()
    if vertexai_attr:
        return gen_ai_attributes.GenAiSystemValues.VERTEX_AI.name.lower()
    return gen_ai_attributes.GenAiSystemValues.GEMINI.name.lower()


def _model_dump_to_tool_definition(tool: Any) -> ToolDefinition:
    model_dump = tool.model_dump(exclude_none=True)

    name = (
        model_dump.get("name")
        or getattr(tool, "name", None)
        or type(tool).__name__
    )
    description = model_dump.get("description") or getattr(
        tool, "description", None
    )
    parameters = model_dump.get("parameters") or model_dump.get("inputSchema")
    return FunctionToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
    )


def _clean_parameters(params: Any) -> Any:
    """Converts parameter objects into plain dicts."""
    if params is None:
        return None
    if isinstance(params, dict):
        return params
    if hasattr(params, "to_dict"):
        return params.to_dict()
    if hasattr(params, "model_dump"):
        return params.model_dump(exclude_none=True)

    try:
        # Check if it's already a standard JSON type.
        json.dumps(params)
        return params

    except (TypeError, ValueError):
        return {
            "type": "object",
            "properties": {
                "serialization_error": {
                    "type": "string",
                    "description": f"Failed to serialize parameters: {type(params).__name__}",
                }
            },
        }


def _tool_to_tool_definition(tool: Tool) -> list[ToolDefinition]:
    definitions = []
    if tool.function_declarations:
        for fd in tool.function_declarations:
            definitions.append(
                FunctionToolDefinition(
                    name=getattr(fd, "name", type(fd).__name__),
                    description=getattr(fd, "description", None),
                    parameters=_clean_parameters(
                        getattr(fd, "parameters", None)
                    ),
                )
            )

    # Generic types
    if hasattr(tool, "model_dump"):
        exclude_fields = {"function_declarations"}
        fields = {
            k: v
            for k, v in tool.model_dump().items()
            if v is not None and k not in exclude_fields
        }

        for tool_type, _ in fields.items():
            definitions.append(
                GenericToolDefinition(
                    type=tool_type,
                    name=tool_type,
                )
            )

    return definitions


def _callable_tool_to_tool_definition(tool: Any) -> ToolDefinition:
    doc = getattr(tool, "__doc__", "") or ""
    return FunctionToolDefinition(
        name=getattr(tool, "__name__", type(tool).__name__),
        description=doc.strip(),
        parameters=None,
    )


def _mcp_tool_to_tool_definition(tool: McpTool) -> ToolDefinition:
    if hasattr(tool, "model_dump"):
        return _model_dump_to_tool_definition(tool)

    return FunctionToolDefinition(
        name=getattr(tool, "name", type(tool).__name__),
        description=getattr(tool, "description", None),
        parameters=getattr(tool, "input_schema", None),
    )


def _to_tool_definition_common(tool: ToolUnionDict) -> list[ToolDefinition]:
    if isinstance(tool, Tool):
        return _tool_to_tool_definition(tool)

    if callable(tool):
        return [_callable_tool_to_tool_definition(tool)]

    if _is_mcp_imported and isinstance(tool, McpTool):
        return [_mcp_tool_to_tool_definition(tool)]

    return [
        GenericToolDefinition(
            name="UnserializableTool",
            type=type(tool).__name__,
        )
    ]


def _to_tool_definition(tool: ToolUnionDict) -> list[ToolDefinition]:
    if _is_mcp_imported and isinstance(tool, McpClientSession):
        return []

    return _to_tool_definition_common(tool)


async def _to_tool_definition_async(
    tool: ToolUnionDict,
) -> list[ToolDefinition]:
    if _is_mcp_imported and isinstance(tool, McpClientSession):
        result = await tool.list_tools()
        return [_model_dump_to_tool_definition(t) for t in result.tools]

    return _to_tool_definition_common(tool)


def _apply_request_attributes(
    config: GenerateContentConfig,
    allow_list: AllowList,
    invocation: InferenceInvocation,
) -> None:
    invocation.temperature = config.temperature
    invocation.top_p = config.top_p
    invocation.top_k = config.top_k
    invocation.request_choice_count = config.candidate_count
    invocation.max_tokens = config.max_output_tokens
    invocation.stop_sequences = config.stop_sequences
    invocation.frequency_penalty = config.frequency_penalty
    invocation.presence_penalty = config.presence_penalty
    invocation.seed = config.seed
    if config.response_mime_type == "text/plain":
        invocation.output_type = "text"
    elif config.response_mime_type == "application/json":
        invocation.output_type = "json"
    else:
        invocation.output_type = config.response_mime_type
    attributes = flatten_dict(
        config.model_dump(exclude_none=True),
        # A custom prefix is used, because the names/structure of the
        # configuration is likely to be specific to Google Gen AI SDK.
        key_prefix=GCP_GENAI_OPERATION_CONFIG,
        # These are all captured above as semantic conventions.
        exclude_keys={
            f"{GCP_GENAI_OPERATION_CONFIG}.system_instruction",
            f"{GCP_GENAI_OPERATION_CONFIG}.temperature",
            f"{GCP_GENAI_OPERATION_CONFIG}.top_k",
            f"{GCP_GENAI_OPERATION_CONFIG}.top_p",
            f"{GCP_GENAI_OPERATION_CONFIG}.candidate_count",
            f"{GCP_GENAI_OPERATION_CONFIG}.max_output_tokens",
            f"{GCP_GENAI_OPERATION_CONFIG}.stop_sequences",
            f"{GCP_GENAI_OPERATION_CONFIG}.frequency_penalty",
            f"{GCP_GENAI_OPERATION_CONFIG}.presence_penalty",
            f"{GCP_GENAI_OPERATION_CONFIG}.seed",
            f"{GCP_GENAI_OPERATION_CONFIG}.response_mime_type",
        },
    )
    invocation.attributes.update(
        {k: v for k, v in attributes.items() if allow_list.allowed(k)}
    )


def _get_response_property(response: GenerateContentResponse, path: str):
    path_segments = path.split(".")
    current_context = response
    for path_segment in path_segments:
        if current_context is None:
            return None
        if isinstance(current_context, dict):
            current_context = current_context.get(path_segment)
        else:
            current_context = getattr(current_context, path_segment)
    return current_context


def _wrapped_config_with_tools(
    telemetry_handler: TelemetryHandler,
    config: GenerateContentConfigOrDict | None,
) -> tuple[GenerateContentConfig, bool]:
    if config is None:
        return GenerateContentConfig(), False
    if isinstance(config, dict):
        try:
            config = GenerateContentConfig.model_validate(config)
        except Exception:
            return GenerateContentConfig(), False
    if not config.tools:
        return config, False
    config = copy(config)
    config.tools = [
        wrapped_tool(tool, telemetry_handler) for tool in config.tools
    ]
    return config, True


def _get_extra_generate_content_attributes() -> dict[str, AttributeValue]:
    attrs = context_api.get_value(
        GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY
    )
    return dict(attrs or {})


def _apply_response_attributes(
    response: GenerateContentResponse,
    finish_reasons: list[str],
    invocation: InferenceInvocation,
):
    if response.response_id:
        invocation.response_id = response.response_id
    if response.model_version:
        invocation.response_model_name = response.model_version
    for candidate in response.candidates or []:
        if candidate.finish_reason:
            finish_reasons.append(candidate.finish_reason.value.lower())
    invocation.finish_reasons = finish_reasons
    input_tokens = _get_response_property(
        response, "usage_metadata.prompt_token_count"
    )
    output_tokens = _get_response_property(
        response, "usage_metadata.candidates_token_count"
    )
    cached_tokens = _get_response_property(
        response, "usage_metadata.cached_content_token_count"
    )
    thinking_tokens = _get_response_property(
        response, "usage_metadata.thoughts_token_count"
    )
    if cached_tokens is not None and isinstance(cached_tokens, int):
        invocation.cache_read_input_tokens = cached_tokens
    if input_tokens is not None and isinstance(input_tokens, int):
        invocation.input_tokens = input_tokens
    if output_tokens is not None and isinstance(output_tokens, int):
        invocation.output_tokens = output_tokens
    if thinking_tokens is not None and isinstance(thinking_tokens, int):
        # The util library will add this total to output tokens.
        invocation.thinking_tokens = thinking_tokens


def _maybe_get_tool_definitions(
    config: GenerateContentConfig,
) -> list[ToolDefinition]:
    return [
        de
        for tool in config.tools or []
        for de in _to_tool_definition(tool)
        if de
    ]


async def _maybe_get_tool_definitions_async(
    config: GenerateContentConfig,
) -> list[ToolDefinition]:
    # Tool discovery is telemetry-only. Do not call MCP/application callbacks
    # before the real SDK operation; losing those definitions is preferable to
    # adding side effects or blocking the customer's request.
    return [
        de
        for tool in config.tools or []
        for de in _to_tool_definition(tool)
        if de
    ]


@dataclass
class _GenerateContentAdviceState:
    invocation: InferenceInvocation
    wrapped_config: GenerateContentConfig
    has_wrapped_tools: bool


@hook_advice("google-genai", "prepare_generate_content")
def _prepare_generate_content_advice(
    telemetry_handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
    instance: Any,
    model: str,
    contents: Union[ContentListUnion, ContentListUnionDict],
    config: Optional[GenerateContentConfigOrDict],
) -> _GenerateContentAdviceState:
    invocation = None
    try:
        wrapped_config, has_wrapped_tools = _wrapped_config_with_tools(
            telemetry_handler,
            config,
        )
        _, server_address = _get_client_info(instance)
        invocation = telemetry_handler.inference(
            provider=_determine_genai_system(instance),
            request_model=model,
            operation_name="generate_content",
            server_address=server_address,
        )
        _apply_request_attributes(
            wrapped_config,
            generate_content_config_key_allowlist,
            invocation,
        )
        invocation.attributes.update(_get_extra_generate_content_attributes())
        invocation.tool_definitions = _maybe_get_tool_definitions(
            wrapped_config
        )
        if telemetry_handler.should_capture_content():
            invocation.input_messages = to_input_messages(
                contents=transformers.t_contents(contents)
            )
            if wrapped_config.system_instruction:
                invocation.system_instruction = to_system_instructions(
                    content=transformers.t_contents(
                        wrapped_config.system_instruction
                    )[0]
                )
        return _GenerateContentAdviceState(
            invocation=invocation,
            wrapped_config=wrapped_config,
            has_wrapped_tools=has_wrapped_tools,
        )
    except Exception:
        if invocation is not None:
            telemetry_handler.abandon_inference(invocation)
        raise


@async_hook_advice("google-genai", "prepare_async_generate_content")
async def _prepare_async_generate_content_advice(
    telemetry_handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
    instance: Any,
    model: str,
    contents: Union[ContentListUnion, ContentListUnionDict],
    config: Optional[GenerateContentConfigOrDict],
) -> _GenerateContentAdviceState:
    invocation = None
    try:
        wrapped_config, has_wrapped_tools = _wrapped_config_with_tools(
            telemetry_handler,
            config,
        )
        _, server_address = _get_client_info(instance)
        invocation = telemetry_handler.inference(
            provider=_determine_genai_system(instance),
            request_model=model,
            operation_name="generate_content",
            server_address=server_address,
        )
        _apply_request_attributes(
            wrapped_config,
            generate_content_config_key_allowlist,
            invocation,
        )
        invocation.attributes.update(_get_extra_generate_content_attributes())
        invocation.tool_definitions = await _maybe_get_tool_definitions_async(
            wrapped_config
        )
        if telemetry_handler.should_capture_content():
            invocation.input_messages = to_input_messages(
                contents=transformers.t_contents(contents)
            )
            if wrapped_config.system_instruction:
                invocation.system_instruction = to_system_instructions(
                    content=transformers.t_contents(
                        wrapped_config.system_instruction
                    )[0]
                )
        return _GenerateContentAdviceState(
            invocation=invocation,
            wrapped_config=wrapped_config,
            has_wrapped_tools=has_wrapped_tools,
        )
    except Exception:
        if invocation is not None:
            telemetry_handler.abandon_inference(invocation)
        raise


@hook_advice("google-genai", "complete_generate_content")
def _complete_generate_content_advice(
    telemetry_handler: TelemetryHandler,
    state: _GenerateContentAdviceState,
    response: GenerateContentResponse,
) -> None:
    finish_reasons: list[str] = []
    try:
        _apply_response_attributes(
            response,
            finish_reasons,
            state.invocation,
        )
        if telemetry_handler.should_capture_content() and response.candidates:
            state.invocation.output_messages = to_output_messages(
                candidates=response.candidates
            )
    finally:
        state.invocation.stop()


@hook_advice("google-genai", "fail_generate_content")
def _fail_generate_content_advice(
    state: _GenerateContentAdviceState,
    error: BaseException,
) -> None:
    state.invocation.fail(error)


@hook_advice("google-genai", "detach_stream_context")
def _detach_stream_context_advice(
    telemetry_handler: TelemetryHandler,
    state: _GenerateContentAdviceState,
) -> bool:
    telemetry_handler.detach_inference_context(state.invocation)
    return True


@hook_advice("google-genai", "abandon_generate_content")
def _abandon_generate_content_advice(
    telemetry_handler: TelemetryHandler,
    state: _GenerateContentAdviceState,
) -> None:
    telemetry_handler.abandon_inference(state.invocation)


def _create_instrumented_generate_content(
    telemetry_handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
):
    def instrumented_generate_content(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ):
        def _execute(
            model: str,
            contents: Union[ContentListUnion, ContentListUnionDict],
            config: Optional[GenerateContentConfigOrDict] = None,
            *_args,
            **_kwargs,
        ):
            state = _prepare_generate_content_advice(
                telemetry_handler,
                generate_content_config_key_allowlist,
                instance,
                model,
                contents,
                config,
            )
            call_config = (
                state.wrapped_config
                if state is not None and state.has_wrapped_tools
                else config
            )
            try:
                response = wrapped(
                    model=model,
                    contents=contents,
                    config=call_config,
                    *_args,
                    **_kwargs,
                )
            except BaseException as error:
                if state is not None:
                    _fail_generate_content_advice(state, error)
                raise
            if state is not None:
                _complete_generate_content_advice(
                    telemetry_handler,
                    state,
                    response,
                )
            return response

        return _execute(*args, **kwargs)

    return instrumented_generate_content


class GenerateContentStreamWrapper(SyncStreamWrapper[GenerateContentResponse]):
    def __init__(
        self,
        stream: Iterable[GenerateContentResponse],
        invocation: InferenceInvocation,
        telemetry_handler: TelemetryHandler,
    ) -> None:
        super().__init__(stream)
        self._self_invocation = invocation
        self._self_telemetry_handler = telemetry_handler
        self._self_finish_reasons: list[str] = []
        self._self_output_accumulator = StreamOutputMessageAccumulator()

    def _process_chunk(self, chunk: GenerateContentResponse) -> None:
        self._self_invocation.record_first_token()
        _apply_response_attributes(
            chunk,
            self._self_finish_reasons,
            self._self_invocation,
        )
        if chunk.candidates:
            self._self_output_accumulator.add_candidates(chunk.candidates)

    def _on_stream_end(self) -> None:
        try:
            if self._self_telemetry_handler.should_capture_content():
                self._self_invocation.output_messages = (
                    self._self_output_accumulator.to_output_messages()
                )
        finally:
            self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


class AsyncGenerateContentStreamWrapper(
    AsyncStreamWrapper[GenerateContentResponse]
):
    def __init__(
        self,
        stream: AsyncIterable[GenerateContentResponse],
        invocation: InferenceInvocation,
        telemetry_handler: TelemetryHandler,
    ) -> None:
        super().__init__(stream)
        # _self_ is a naming convention used by the wrapt library to differentiate
        # between attributes on the wrapped function and the original function.
        self._self_invocation = invocation
        self._self_telemetry_handler = telemetry_handler
        self._self_finish_reasons: list[str] = []
        self._self_output_accumulator = StreamOutputMessageAccumulator()

    def _process_chunk(self, chunk: GenerateContentResponse) -> None:
        self._self_invocation.record_first_token()
        _apply_response_attributes(
            chunk,
            self._self_finish_reasons,
            self._self_invocation,
        )
        if chunk.candidates:
            self._self_output_accumulator.add_candidates(chunk.candidates)

    def _on_stream_end(self) -> None:
        try:
            if self._self_telemetry_handler.should_capture_content():
                self._self_invocation.output_messages = (
                    self._self_output_accumulator.to_output_messages()
                )
        finally:
            self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


@hook_advice("google-genai", "wrap_generate_content_stream")
def _wrap_generate_content_stream_advice(
    stream: Any,
    state: _GenerateContentAdviceState,
    telemetry_handler: TelemetryHandler,
    *,
    asynchronous: bool,
) -> Any:
    wrapper_type = (
        AsyncGenerateContentStreamWrapper
        if asynchronous
        else GenerateContentStreamWrapper
    )
    return wrapper_type(
        stream,
        state.invocation,
        telemetry_handler,
    )


def _create_instrumented_generate_content_stream(
    telemetry_handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
):
    def instrumented_generate_content_stream(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ):
        def _execute(
            model: str,
            contents: Union[ContentListUnion, ContentListUnionDict],
            config: Optional[GenerateContentConfigOrDict] = None,
            *_args,
            **_kwargs,
        ):
            state = _prepare_generate_content_advice(
                telemetry_handler,
                generate_content_config_key_allowlist,
                instance,
                model,
                contents,
                config,
            )
            call_config = (
                state.wrapped_config
                if state is not None and state.has_wrapped_tools
                else config
            )
            try:
                stream = wrapped(
                    model=model,
                    contents=contents,
                    config=call_config,
                    *_args,
                    **_kwargs,
                )
            except BaseException as error:
                if state is not None:
                    _fail_generate_content_advice(state, error)
                raise
            if state is None:
                return stream
            if not _detach_stream_context_advice(
                telemetry_handler,
                state,
            ):
                _abandon_generate_content_advice(
                    telemetry_handler,
                    state,
                )
                return stream
            stream_wrapper = _wrap_generate_content_stream_advice(
                stream,
                state,
                telemetry_handler,
                asynchronous=False,
            )
            if stream_wrapper is None:
                _abandon_generate_content_advice(
                    telemetry_handler,
                    state,
                )
                return stream
            return stream_wrapper

        return _execute(*args, **kwargs)

    return instrumented_generate_content_stream


def _create_instrumented_async_generate_content(
    telemetry_handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
):
    async def instrumented_generate_content(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ):
        async def _execute(
            model: str,
            contents: Union[ContentListUnion, ContentListUnionDict],
            config: Optional[GenerateContentConfigOrDict] = None,
            *_args,
            **_kwargs,
        ):
            state = await _prepare_async_generate_content_advice(
                telemetry_handler,
                generate_content_config_key_allowlist,
                instance,
                model,
                contents,
                config,
            )
            call_config = (
                state.wrapped_config
                if state is not None and state.has_wrapped_tools
                else config
            )
            try:
                response = await wrapped(
                    model=model,
                    contents=contents,
                    config=call_config,
                    *_args,
                    **_kwargs,
                )
            except BaseException as error:
                if state is not None:
                    _fail_generate_content_advice(state, error)
                raise
            if state is not None:
                _complete_generate_content_advice(
                    telemetry_handler,
                    state,
                    response,
                )
            return response

        return await _execute(*args, **kwargs)

    return instrumented_generate_content


def _create_instrumented_async_generate_content_stream(  # type: ignore
    telemetry_handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
):
    async def instrumented_generate_content_stream(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ):
        async def _execute(
            model: str,
            contents: Union[ContentListUnion, ContentListUnionDict],
            config: Optional[GenerateContentConfigOrDict] = None,
            *_args,
            **_kwargs,
        ):
            state = await _prepare_async_generate_content_advice(
                telemetry_handler,
                generate_content_config_key_allowlist,
                instance,
                model,
                contents,
                config,
            )
            call_config = (
                state.wrapped_config
                if state is not None and state.has_wrapped_tools
                else config
            )
            try:
                stream = await wrapped(
                    model=model,
                    contents=contents,
                    config=call_config,
                    *_args,
                    **_kwargs,
                )
            except BaseException as error:
                if state is not None:
                    _fail_generate_content_advice(state, error)
                raise
            if state is None:
                return stream
            if not _detach_stream_context_advice(
                telemetry_handler,
                state,
            ):
                _abandon_generate_content_advice(
                    telemetry_handler,
                    state,
                )
                return stream
            stream_wrapper = _wrap_generate_content_stream_advice(
                stream,
                state,
                telemetry_handler,
                asynchronous=True,
            )
            if stream_wrapper is None:
                _abandon_generate_content_advice(
                    telemetry_handler,
                    state,
                )
                return stream
            return stream_wrapper

        return await _execute(*args, **kwargs)

    return instrumented_generate_content_stream


def uninstrument_generate_content(snapshot: object):
    assert isinstance(snapshot, _MethodsSnapshot)
    snapshot.restore()


def instrument_generate_content(
    telemetry_handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
) -> object:
    os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT", "true")
    snapshot = _MethodsSnapshot()
    wrapped = wrap_function_wrapper(
        "google.genai.models",
        "Models.generate_content",
        _create_instrumented_generate_content(
            telemetry_handler,
            generate_content_config_key_allowlist,
        ),
    )
    wrapped2 = wrap_function_wrapper(
        "google.genai.models",
        "Models.generate_content_stream",
        _create_instrumented_generate_content_stream(
            telemetry_handler,
            generate_content_config_key_allowlist,
        ),
    )
    wrapped3 = wrap_function_wrapper(
        "google.genai.models",
        "AsyncModels.generate_content",
        _create_instrumented_async_generate_content(
            telemetry_handler,
            generate_content_config_key_allowlist,
        ),
    )
    wrapped4 = wrap_function_wrapper(
        "google.genai.models",
        "AsyncModels.generate_content_stream",
        _create_instrumented_async_generate_content_stream(
            telemetry_handler,
            generate_content_config_key_allowlist,
        ),
    )
    _set_co_filename(wrapped)
    _set_co_filename(wrapped2)
    _set_co_filename(wrapped3)
    _set_co_filename(wrapped4)
    return snapshot
