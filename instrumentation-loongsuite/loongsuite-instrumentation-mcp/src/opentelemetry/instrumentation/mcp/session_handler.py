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

import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Mapping, Tuple, Union, cast
from urllib.parse import urlparse

import wrapt

from opentelemetry import propagate
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.mcp.metrics import ServerMetrics
from opentelemetry.instrumentation.mcp.semconv import (
    MCPAttributes,
    _method_names_with_target,
)
from opentelemetry.instrumentation.mcp.utils import _get_logger
from opentelemetry.trace import SpanKind

has_mcp_types = False
try:
    from mcp.shared.message import SessionMessage
    from mcp.shared.session import RequestResponder
    from mcp.types import ClientRequest, JSONRPCRequest, ServerResult

    has_mcp_types = True
except Exception:
    pass

logger = _get_logger(__name__)


class _SessionContext:
    def __init__(self, session_attributes: Dict[str, str]):
        self._attributes = session_attributes
        self._session_id_callback = None

    def _set_session_id_callback(
        self, session_id_callback: Callable[[], Union[str, None]]
    ):
        self._session_id_callback = session_id_callback

    def _parse_session_id(self) -> Union[str, None]:
        if self._session_id_callback is None:
            return None
        if MCPAttributes.MCP_SESSION_ID in self._attributes:
            return self._attributes[MCPAttributes.MCP_SESSION_ID]

        try:
            session_id = self._session_id_callback()
            if session_id:
                self._attributes[MCPAttributes.MCP_SESSION_ID] = str(
                    session_id
                )
                return str(session_id)
        except Exception:
            logger.debug("Failed to get session id", exc_info=True)
        return None

    def _get_session_attributes(self):
        return self._attributes

    def _safe_attach(self, context: Any):
        try:
            if context is None:
                return
            setattr(context[0], "_session_context", self)
        except Exception:
            logger.debug("Failed to attach session context", exc_info=True)

    def _safe_detach(self, context: Any):
        try:
            if context is None or not hasattr(context[0], "_session_context"):
                return
            delattr(context[0], "_session_context")
        except Exception:
            logger.debug("Failed to detach session context", exc_info=True)


def _parse_url(
    args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Tuple[Union[str, None], Union[str, None]]:
    url = None
    if len(args) > 0:
        url = args[0]
    elif "url" in kwargs:
        url = kwargs["url"]
    if url is None or not isinstance(url, str) or url == "":
        return None, None
    try:
        parsed_url = urlparse(url)
        return str(parsed_url.hostname), str(parsed_url.port)
    except Exception:
        logger.debug("Failed to parse url for mcp transport", exc_info=True)
        return None, None


def _get_session_context(
    transport: str, args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> _SessionContext:
    input_attributes = {
        MCPAttributes.COMPONENT_NAME: MCPAttributes.MCP_CLIENT,
        MCPAttributes.NETWORK_TRANSPORT: transport,
    }
    if transport != "stdio":
        try:
            hostname, port = _parse_url(args, kwargs)
            if hostname is not None:
                input_attributes[MCPAttributes.SERVER_ADDRESS] = hostname
            if port is not None:
                input_attributes[MCPAttributes.SERVER_PORT] = port
        except Exception:
            logger.debug("Failed to set span attributes", exc_info=True)
    return _SessionContext(input_attributes)


# do propagation in send method
async def _writer_send_wrapper(
    wrapped: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
):
    try:
        if len(args) > 0 and has_mcp_types:
            session_message = args[0]
            if isinstance(session_message, SessionMessage) and isinstance(
                session_message.message.root, JSONRPCRequest
            ):
                request = session_message.message.root
                if request.params is None:
                    request.params = {}
                meta = request.params.setdefault("_meta", {})
                propagate.get_global_textmap().inject(meta)
    except Exception:
        logger.debug("Failed to propagate", exc_info=True)

    return await wrapped(*args, **kwargs)


def _safe_propagate(context: Any):
    try:
        writer = context[1]
        if hasattr(writer, "send"):
            wrapt.wrap_function_wrapper(writer, "send", _writer_send_wrapper)
    except Exception:
        logger.debug("Failed to propagate", exc_info=True)


def sse_client_wrapper():
    @asynccontextmanager
    async def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ):
        session_context = _get_session_context("sse", args, kwargs)
        async with wrapped(*args, **kwargs) as context:
            _safe_propagate(context)
            session_context._safe_attach(context)
            try:
                yield context
            finally:
                session_context._safe_detach(context)

    return wrapper


def websocket_client_wrapper():
    @asynccontextmanager
    async def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ):
        session_context = _get_session_context("websocket", args, kwargs)
        async with wrapped(*args, **kwargs) as context:
            _safe_propagate(context)
            session_context._safe_attach(context)
            try:
                yield context
            finally:
                session_context._safe_detach(context)

    return wrapper


def stdio_client_wrapper():
    @asynccontextmanager
    async def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ):
        session_context = _get_session_context("stdio", args, kwargs)
        async with wrapped(*args, **kwargs) as context:
            _safe_propagate(context)
            session_context._safe_attach(context)
            try:
                yield context
            finally:
                session_context._safe_detach(context)

    return wrapper


def streamable_http_client_wrapper():
    @asynccontextmanager
    async def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ):
        session_context = _get_session_context("http", args, kwargs)
        async with wrapped(*args, **kwargs) as context:
            try:
                session_context._set_session_id_callback(context[2])
            except Exception:
                logger.debug(
                    "Failed to get session id callback", exc_info=True
                )
            _safe_propagate(context)
            session_context._safe_attach(context)
            try:
                yield context
            finally:
                session_context._safe_detach(context)

    return wrapper


class ServerHandleRequestWrapper:
    def __init__(
        self, tracer: trace_api.Tracer, server_metrics: ServerMetrics
    ):
        self._tracer = tracer
        self._server_metrics = server_metrics

    def extract_parent_context(self, args: Tuple[Any, ...]) -> Any:
        if not has_mcp_types:
            return None
        try:
            if len(args) > 0:
                message = cast(
                    RequestResponder[ClientRequest, ServerResult], args[0]
                )
                carrier = getattr(
                    message.request_meta, "__pydantic_extra__", None
                )
                if carrier:
                    return propagate.extract(carrier)
        except Exception:
            logger.debug("Failed to extract trace context", exc_info=True)
        return None

    def extract_attributes(
        self, args: Tuple[Any, ...]
    ) -> Tuple[Dict[str, str], Union[str, None]]:
        attributes: Dict[str, str] = {}
        span_name = None
        if not has_mcp_types:
            return attributes, span_name

        try:
            if len(args) == 0:
                return attributes, None

            message = cast(
                RequestResponder[ClientRequest, ServerResult], args[0]
            )
            method = message.request.root.method
            if method is None:
                return attributes, None

            attributes[MCPAttributes.RPC_REQUEST_ID] = str(message.request_id)
            attributes[MCPAttributes.MCP_METHOD_NAME] = method

            # span name
            if method in ["tools/call", "prompts/get"]:
                target_name = message.request.root.params.name  # type: ignore
                span_name = f"{method} {target_name}"
            elif method in [
                "resources/read",
                "resources/subscribe",
                "resources/unsubscribe",
            ]:
                target_name = str(message.request.root.params.uri)  # type: ignore
                span_name = f"{method} {target_name}"
            else:
                span_name = method
                target_name = None

            if target_name is not None and method in _method_names_with_target:
                (attr_name, _) = _method_names_with_target[method]
                attributes[attr_name] = target_name

        except Exception:
            logger.debug("Failed to extract attributes", exc_info=True)
        return attributes, span_name

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ):
        start_time = time.time()
        parent_context = self.extract_parent_context(args)
        attributes, span_name = self.extract_attributes(args)

        if span_name is None:
            return await wrapped(*args, **kwargs)

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.SERVER,
            context=parent_context,
            attributes=attributes,
        ):
            try:
                return await wrapped(*args, **kwargs)
            finally:
                try:
                    self._server_metrics.operation_duration.record(
                        time.time() - start_time
                    )
                    self._server_metrics.operation_count.add(1)
                except Exception:
                    logger.debug(
                        "Failed to record server operation duration",
                        exc_info=True,
                    )
