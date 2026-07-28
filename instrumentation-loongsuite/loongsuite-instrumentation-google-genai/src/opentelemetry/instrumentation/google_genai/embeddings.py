# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from google.genai._api_client import BaseApiClient
from google.genai.models import AsyncModels, Models
from google.genai.types import EmbedContentResponse
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.google_genai.client_info import (
    get_client_info as _get_client_info,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai import hook_advice

from ._compat import EmbeddingInvocation, TelemetryHandler

_RAW_RESPONSE_BODY: ContextVar[str | None] = ContextVar(
    "raw_response_body", default=None
)
_CAPTURE_RAW_RESPONSE: ContextVar[bool] = ContextVar(
    "capture_raw_response", default=False
)
_EMBEDDING_DIMENSION_COUNT = getattr(
    GenAIAttributes,
    "GEN_AI_EMBEDDINGS_DIMENSION_COUNT",
    "gen_ai.embeddings.dimension.count",
)


class _EmbeddingMethodsSnapshot:
    def __init__(self) -> None:
        self._original_embed_content = Models.embed_content
        self._original_embed_content_code = Models.embed_content.__code__
        self._original_async_embed_content = AsyncModels.embed_content
        self._original_async_embed_content_code = (
            AsyncModels.embed_content.__code__
        )
        self._original_client_request = BaseApiClient.request
        self._original_client_async_request = BaseApiClient.async_request

    def restore(self) -> None:
        self._original_embed_content.__code__ = (
            self._original_embed_content_code
        )
        self._original_async_embed_content.__code__ = (
            self._original_async_embed_content_code
        )

        Models.embed_content = self._original_embed_content
        AsyncModels.embed_content = self._original_async_embed_content
        BaseApiClient.request = self._original_client_request
        BaseApiClient.async_request = self._original_client_async_request


# Magic incantation used by native Google ADK instrumentation to identify
# instrumented functions and suppress its own internal tracing when OTel is active.
def _set_co_filename(wrapped: object) -> None:
    wrapped.__wrapped__.__code__ = wrapped.__wrapped__.__code__.replace(
        co_filename=__file__.replace("\\", "/")
    )


def _apply_embedding_response_attributes(
    response: EmbedContentResponse,
    invocation: EmbeddingInvocation,
) -> None:
    if response.embeddings:
        first_embedding = response.embeddings[0]
        if first_embedding.values:
            invocation.dimension_count = len(first_embedding.values)
            invocation.metric_attributes[_EMBEDDING_DIMENSION_COUNT] = (
                invocation.dimension_count
            )

    # In the future we can get rid of this and the monkey patching of the
    # requests, and use the parsed SDK response instead. See:
    # https://github.com/googleapis/python-genai/issues/2658
    if raw_body := _RAW_RESPONSE_BODY.get():
        try:
            body_dict = json.loads(raw_body)
            usage_metadata = body_dict.get("usageMetadata")
            if isinstance(usage_metadata, dict):
                invocation.input_tokens = usage_metadata.get(
                    "promptTokenCount"
                )
        except Exception:
            pass


@dataclass
class _EmbeddingAdviceState:
    invocation: EmbeddingInvocation
    raw_body_token: Any
    capture_token: Any


@hook_advice("google-genai", "prepare_embedding")
def _prepare_embedding_advice(
    telemetry_handler: TelemetryHandler,
    instance: Models | AsyncModels,
    kwargs: dict[str, Any],
) -> _EmbeddingAdviceState:
    raw_body_token = None
    capture_token = None
    invocation = None
    try:
        raw_body_token = _RAW_RESPONSE_BODY.set(None)
        capture_token = _CAPTURE_RAW_RESPONSE.set(True)
        is_vertex, server_address = _get_client_info(instance)
        invocation = telemetry_handler.embedding(
            provider=(
                GenAIAttributes.GenAiSystemValues.VERTEX_AI.value
                if is_vertex
                else GenAIAttributes.GenAiSystemValues.GEMINI.value
            ),
            request_model=kwargs.get("model"),
            server_address=server_address,
        )
        return _EmbeddingAdviceState(
            invocation=invocation,
            raw_body_token=raw_body_token,
            capture_token=capture_token,
        )
    except Exception:
        if invocation is not None:
            telemetry_handler.abandon_embedding(invocation)
        if raw_body_token is not None:
            _RAW_RESPONSE_BODY.reset(raw_body_token)
        if capture_token is not None:
            _CAPTURE_RAW_RESPONSE.reset(capture_token)
        raise


@hook_advice("google-genai", "complete_embedding")
def _complete_embedding_advice(
    state: _EmbeddingAdviceState,
    response: EmbedContentResponse,
) -> None:
    try:
        _apply_embedding_response_attributes(response, state.invocation)
        state.invocation.stop()
    finally:
        _RAW_RESPONSE_BODY.reset(state.raw_body_token)
        _CAPTURE_RAW_RESPONSE.reset(state.capture_token)


@hook_advice("google-genai", "fail_embedding")
def _fail_embedding_advice(
    state: _EmbeddingAdviceState,
    error: BaseException,
) -> None:
    try:
        state.invocation.fail(error)
    finally:
        _RAW_RESPONSE_BODY.reset(state.raw_body_token)
        _CAPTURE_RAW_RESPONSE.reset(state.capture_token)


@hook_advice("google-genai", "capture_embedding_raw_response")
def _capture_embedding_raw_response_advice(response: Any) -> None:
    if (
        _CAPTURE_RAW_RESPONSE.get()
        and response
        and getattr(response, "body", None)
    ):
        _RAW_RESPONSE_BODY.set(response.body)


def _create_instrumented_embed_content(
    telemetry_handler: TelemetryHandler,
) -> Callable[
    [
        Callable[..., EmbedContentResponse],
        Models,
        tuple[Any, ...],
        dict[str, Any],
    ],
    EmbedContentResponse,
]:
    def instrumented_embed_content(
        wrapped: Callable[..., EmbedContentResponse],
        instance: Models,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> EmbedContentResponse:
        state = _prepare_embedding_advice(
            telemetry_handler,
            instance,
            kwargs,
        )
        try:
            response = wrapped(*args, **kwargs)
        except BaseException as error:
            if state is not None:
                _fail_embedding_advice(state, error)
            raise
        if state is not None:
            _complete_embedding_advice(state, response)
        return response

    return instrumented_embed_content


def _create_instrumented_async_embed_content(
    telemetry_handler: TelemetryHandler,
) -> Callable[
    [
        Callable[..., Any],
        AsyncModels,
        tuple[Any, ...],
        dict[str, Any],
    ],
    Any,
]:
    async def instrumented_embed_content(
        wrapped: Callable[..., Any],
        instance: AsyncModels,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> EmbedContentResponse:
        state = _prepare_embedding_advice(
            telemetry_handler,
            instance,
            kwargs,
        )
        try:
            response = await wrapped(*args, **kwargs)
        except BaseException as error:
            if state is not None:
                _fail_embedding_advice(state, error)
            raise
        if state is not None:
            _complete_embedding_advice(state, response)
        return response

    return instrumented_embed_content


def uninstrument_embeddings(snapshot: object) -> None:
    assert isinstance(snapshot, _EmbeddingMethodsSnapshot)
    snapshot.restore()


def instrument_embeddings(
    telemetry_handler: TelemetryHandler,
) -> object:
    snapshot = _EmbeddingMethodsSnapshot()

    wrapped = wrap_function_wrapper(
        "google.genai.models",
        "Models.embed_content",
        _create_instrumented_embed_content(telemetry_handler),
    )
    wrapped2 = wrap_function_wrapper(
        "google.genai.models",
        "AsyncModels.embed_content",
        _create_instrumented_async_embed_content(telemetry_handler),
    )
    _set_co_filename(wrapped)
    _set_co_filename(wrapped2)

    # Wrap BaseApiClient to capture raw responses
    def instrumented_request(wrapped, instance, args, kwargs):
        response = wrapped(*args, **kwargs)
        _capture_embedding_raw_response_advice(response)
        return response

    async def instrumented_async_request(wrapped, instance, args, kwargs):
        response = await wrapped(*args, **kwargs)
        _capture_embedding_raw_response_advice(response)
        return response

    wrap_function_wrapper(
        "google.genai._api_client",
        "BaseApiClient.request",
        instrumented_request,
    )
    wrap_function_wrapper(
        "google.genai._api_client",
        "BaseApiClient.async_request",
        instrumented_async_request,
    )

    return snapshot
