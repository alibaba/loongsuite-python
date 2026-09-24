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

"""
Core wrappers for Mem0 instrumentation.
Implements wrapping logic for top-level Memory operations and sub-phase operations.
"""

import inspect
import logging
from functools import wraps
from typing import Any, Callable, Optional

from opentelemetry.instrumentation.mem0.config import (
    is_internal_phases_enabled,
)
from opentelemetry.instrumentation.mem0.internal._extractors import (
    GraphOperationAttributeExtractor,
    MemoryOperationAttributeExtractor,
    RerankerAttributeExtractor,
    VectorOperationAttributeExtractor,
)
from opentelemetry.instrumentation.mem0.internal._util import (
    extract_server_info,
    get_exception_type,
    safe_str,
)
from opentelemetry.instrumentation.mem0.semconv import (
    SemanticAttributes,
    SpanName,
)
from opentelemetry.instrumentation.mem0.types import (
    HookContext,
    InnerAfterHook,
    InnerBeforeHook,
    MemoryAfterHook,
    MemoryBeforeHook,
    safe_call_hook,
)
from opentelemetry.trace import (
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    get_current_span,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_memory import MemoryInvocation
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)


def _get_field(payload: dict, field_name: str) -> Any:
    """
    fetch a field if present (distinguish between missing vs present None).
    """
    if field_name in payload:
        return payload.get(field_name)
    return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _normalize_call_parameters(
    func: Callable,
    args: tuple,
    kwargs: dict,
) -> dict:
    """
    Generically merge positional and keyword arguments into complete kwargs dict.

    Uses inspect.signature to get function signature and automatically map args to parameter names.
    This method requires no mapping table per operation and adapts automatically to any new method.

    Strategy:
    1. Use inspect.signature to get function signature
    2. Map args to parameter names in order (skip self/cls)
    3. Merge with existing kwargs (kwargs takes priority, no overwrite)
    4. Return complete parameter dict

    Args:
        func: Function/method being called
        args: Positional arguments tuple
        kwargs: Keyword arguments dict

    Returns:
        Normalized complete parameter dict

    Examples:
        >>> def update(self, memory_id, data):
        ...     pass
        >>> _normalize_call_parameters(update, ('id123', 'new data'), {})
        {'memory_id': 'id123', 'data': 'new data'}

        >>> def add(self, messages, *, user_id=None):
        ...     pass
        >>> _normalize_call_parameters(add, ('msg',), {'user_id': 'u1'})
        {'messages': 'msg', 'user_id': 'u1'}
    """
    normalized = dict(kwargs)

    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        # Skip self/cls parameter (usually first parameter)
        start_index = 0
        if params and params[0].name in ("self", "cls"):
            start_index = 1

        # Map args to parameter names
        for idx, arg_value in enumerate(args):
            param_idx = start_index + idx

            # Check if exceeds parameter list
            if param_idx >= len(params):
                break

            param = params[param_idx]

            # Skip *args and **kwargs type parameters
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            param_name = param.name

            # Only add if parameter not already in kwargs (kwargs takes priority)
            if param_name not in normalized:
                normalized[param_name] = arg_value

    except Exception as e:
        logger.debug(f"Failed to normalize call parameters: {e}")

    return normalized


class MemoryOperationWrapper:
    """Memory top-level operation wrapper."""

    def __init__(self, telemetry_handler: ExtendedTelemetryHandler):
        """
        Initialize wrapper.

        Args:
            telemetry_handler: GenAI ExtendedTelemetryHandler from opentelemetry-util-genai.
        """
        self.telemetry_handler = telemetry_handler
        self.extractor = MemoryOperationAttributeExtractor()
        self._memory_before_hook: MemoryBeforeHook = None
        self._memory_after_hook: MemoryAfterHook = None

    def wrap_operation(
        self,
        operation_name: str,
        extract_attributes_func: Optional[Callable] = None,
        is_memory_client: bool = False,
    ) -> Callable:
        """
        Wrap Memory operation method.

        Args:
            operation_name: Operation name (e.g. 'add', 'search')
            extract_attributes_func: Attribute extraction function
            is_memory_client: Whether MemoryClient/AsyncMemoryClient call

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def sync_wrapper(instance: Any, *args, **kwargs):
                return self._execute_with_handler(
                    func,
                    instance,
                    args,
                    kwargs,
                    operation_name,
                    extract_attributes_func,
                    is_memory_client=is_memory_client,
                )

            @wraps(func)
            async def async_wrapper(instance: Any, *args, **kwargs):
                return await self._execute_with_handler_async(
                    func,
                    instance,
                    args,
                    kwargs,
                    operation_name,
                    extract_attributes_func,
                    is_memory_client=is_memory_client,
                )

            # Return corresponding wrapper based on whether original function is coroutine
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    @staticmethod
    def _set_invocation_fields_from_kwargs(
        invocation: MemoryInvocation, normalized_kwargs: dict
    ) -> None:
        """Populate MemoryInvocation standard fields from normalized kwargs."""

        def _int_or_none(value: Any) -> Optional[int]:
            try:
                if value is None:
                    return None
                return int(value)
            except Exception:
                return None

        def _float_or_none(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                return float(value)
            except Exception:
                return None

        if (user_id := normalized_kwargs.get("user_id")) is not None:
            invocation.user_id = safe_str(user_id)
        if (agent_id := normalized_kwargs.get("agent_id")) is not None:
            invocation.agent_id = safe_str(agent_id)
        if (run_id := normalized_kwargs.get("run_id")) is not None:
            invocation.run_id = safe_str(run_id)
        if (app_id := normalized_kwargs.get("app_id")) is not None:
            invocation.app_id = safe_str(app_id)

        if (memory_id := normalized_kwargs.get("memory_id")) is not None:
            invocation.memory_id = safe_str(memory_id)
        if (limit := normalized_kwargs.get("limit")) is not None:
            invocation.limit = _int_or_none(limit)
        if (page := normalized_kwargs.get("page")) is not None:
            invocation.page = _int_or_none(page)
        if (page_size := normalized_kwargs.get("page_size")) is not None:
            invocation.page_size = _int_or_none(page_size)
        if (top_k := normalized_kwargs.get("top_k")) is not None:
            invocation.top_k = _int_or_none(top_k)
        if (memory_type := normalized_kwargs.get("memory_type")) is not None:
            invocation.memory_type = safe_str(memory_type)
        if (threshold := normalized_kwargs.get("threshold")) is not None:
            invocation.threshold = _float_or_none(threshold)
        if "rerank" in normalized_kwargs:
            # rerank can be explicitly False
            raw_rerank = normalized_kwargs.get("rerank")
            invocation.rerank = (
                bool(raw_rerank) if raw_rerank is not None else None
            )

    @staticmethod
    def _apply_custom_extractor_output_to_invocation(
        invocation: MemoryInvocation, extracted: dict
    ) -> None:
        """
        Apply custom extractor output directly onto MemoryInvocation.

        Expected format (no compatibility guarantees):
        - Any MemoryInvocation field name can be provided directly, e.g.:
          user_id/agent_id/run_id/app_id/memory_id/limit/page/page_size/top_k/
          memory_type/threshold/rerank/input_messages/output_messages/
          server_address/server_port
        - Custom attributes can be provided as:
          - key "attributes": dict[str, Any]
          - or any other leftover keys will be treated as custom attributes
        """
        if "user_id" in extracted:
            raw = _get_field(extracted, "user_id")
            invocation.user_id = safe_str(raw) if raw is not None else None
        if "agent_id" in extracted:
            raw = _get_field(extracted, "agent_id")
            invocation.agent_id = safe_str(raw) if raw is not None else None
        if "run_id" in extracted:
            raw = _get_field(extracted, "run_id")
            invocation.run_id = safe_str(raw) if raw is not None else None
        if "app_id" in extracted:
            raw = _get_field(extracted, "app_id")
            invocation.app_id = safe_str(raw) if raw is not None else None
        if "memory_id" in extracted:
            raw = _get_field(extracted, "memory_id")
            invocation.memory_id = safe_str(raw) if raw is not None else None

        if "limit" in extracted:
            invocation.limit = _coerce_optional_int(
                _get_field(extracted, "limit")
            )
        if "page" in extracted:
            invocation.page = _coerce_optional_int(
                _get_field(extracted, "page")
            )
        if "page_size" in extracted:
            invocation.page_size = _coerce_optional_int(
                _get_field(extracted, "page_size")
            )
        if "top_k" in extracted:
            invocation.top_k = _coerce_optional_int(
                _get_field(extracted, "top_k")
            )

        if "memory_type" in extracted:
            raw = _get_field(extracted, "memory_type")
            invocation.memory_type = safe_str(raw) if raw is not None else None
        if "threshold" in extracted:
            invocation.threshold = _coerce_optional_float(
                _get_field(extracted, "threshold")
            )
        if "rerank" in extracted:
            raw_rerank = _get_field(extracted, "rerank")
            invocation.rerank = (
                bool(raw_rerank) if raw_rerank is not None else None
            )

        if "input_messages" in extracted:
            invocation.input_messages = _get_field(extracted, "input_messages")
        if "output_messages" in extracted:
            invocation.output_messages = _get_field(
                extracted, "output_messages"
            )

        if "server_address" in extracted:
            raw = _get_field(extracted, "server_address")
            invocation.server_address = (
                safe_str(raw) if raw is not None else None
            )
        if "server_port" in extracted:
            invocation.server_port = _coerce_optional_int(
                _get_field(extracted, "server_port")
            )

        # Custom attributes
        attrs = extracted.get("attributes")
        if isinstance(attrs, dict):
            invocation.attributes.update(attrs)

        # Any leftover keys -> invocation.attributes (excluding known fields)
        known = {
            "user_id",
            "agent_id",
            "run_id",
            "app_id",
            "memory_id",
            "limit",
            "page",
            "page_size",
            "top_k",
            "memory_type",
            "threshold",
            "rerank",
            "input_messages",
            "output_messages",
            "server_address",
            "server_port",
            "attributes",
        }
        for k, v in extracted.items():
            if k in known:
                continue
            invocation.attributes[k] = v

    def _apply_extracted_attrs_to_invocation(
        self,
        invocation: MemoryInvocation,
        instance: Any,
        normalized_kwargs: dict,
        operation_name: str,
        *,
        result: Any = None,
        extract_attributes_func: Optional[Callable] = None,
        is_memory_client: bool = False,
    ) -> None:
        """Extract attributes using existing Mem0 extractor and map them onto MemoryInvocation."""
        try:
            if extract_attributes_func:
                operation_attrs = extract_attributes_func(
                    normalized_kwargs, result
                )
                if isinstance(operation_attrs, dict):
                    self._apply_custom_extractor_output_to_invocation(
                        invocation, operation_attrs
                    )
                return

            # Built-in extractor path: directly extract for MemoryInvocation fields
            input_msg, output_msg = self.extractor.extract_invocation_content(
                operation_name,
                normalized_kwargs,
                result,
                is_memory_client=is_memory_client,
            )
            if input_msg is not None:
                invocation.input_messages = input_msg
            if output_msg is not None:
                invocation.output_messages = output_msg

            # Extra attributes only (NOT covered by MemoryInvocation fields)
            extra_attrs = self.extractor.extract_invocation_attributes(
                operation_name, normalized_kwargs, result
            )
            if extra_attrs:
                invocation.attributes.update(extra_attrs)
        except Exception as e:
            logger.debug(f"Failed to extract invocation attributes: {e}")

    def _execute_with_handler(
        self,
        func: Callable,
        instance: Any,
        args: tuple,
        kwargs: dict,
        operation_name: str,
        extract_attributes_func: Optional[Callable],
        is_memory_client: bool = False,
    ) -> Any:
        """Top-level Memory operation execution using util GenAI memory handler (sync)."""
        normalized_kwargs = _normalize_call_parameters(func, args, kwargs)

        invocation = MemoryInvocation(operation=operation_name)
        self._set_invocation_fields_from_kwargs(invocation, normalized_kwargs)

        # Server info (MemoryClient/AsyncMemoryClient)
        try:
            address, port = extract_server_info(instance)
            if address:
                invocation.server_address = address
            if port is not None:
                invocation.server_port = port
        except Exception as e:
            logger.debug(f"Failed to extract server info: {e}")

        # Pre-extract request attributes/content (no result yet)
        self._apply_extracted_attrs_to_invocation(
            invocation,
            instance,
            normalized_kwargs,
            operation_name,
            result=None,
            extract_attributes_func=extract_attributes_func,
            is_memory_client=is_memory_client,
        )

        self.telemetry_handler.start_memory(invocation)
        hook_context: HookContext = {}
        # Read current span after util handler starts memory (span should exist in most cases)
        span = get_current_span()
        safe_call_hook(
            self._memory_before_hook,
            span,
            operation_name,
            instance,
            args,
            dict(kwargs),
            hook_context,
        )
        try:
            result = func(*args, **kwargs)
            # Post-extract result attributes/content (must happen before stop_memory)
            self._apply_extracted_attrs_to_invocation(
                invocation,
                instance,
                normalized_kwargs,
                operation_name,
                result=result,
                extract_attributes_func=extract_attributes_func,
                is_memory_client=is_memory_client,
            )
            safe_call_hook(
                self._memory_after_hook,
                span,
                operation_name,
                instance,
                args,
                dict(kwargs),
                hook_context,
                result,
                None,
            )
            self.telemetry_handler.stop_memory(invocation)
            return result
        except Exception as e:
            safe_call_hook(
                self._memory_after_hook,
                span,
                operation_name,
                instance,
                args,
                dict(kwargs),
                hook_context,
                None,
                e,
            )
            self.telemetry_handler.fail_memory(
                invocation, Error(message=str(e), type=type(e))
            )
            raise

    async def _execute_with_handler_async(
        self,
        func: Callable,
        instance: Any,
        args: tuple,
        kwargs: dict,
        operation_name: str,
        extract_attributes_func: Optional[Callable],
        is_memory_client: bool = False,
    ) -> Any:
        """Top-level Memory operation execution using util GenAI memory handler (async)."""
        normalized_kwargs = _normalize_call_parameters(func, args, kwargs)

        invocation = MemoryInvocation(operation=operation_name)
        self._set_invocation_fields_from_kwargs(invocation, normalized_kwargs)

        # Server info (MemoryClient/AsyncMemoryClient)
        try:
            address, port = extract_server_info(instance)
            if address:
                invocation.server_address = address
            if port is not None:
                invocation.server_port = port
        except Exception as e:
            logger.debug(f"Failed to extract server info: {e}")

        # Pre-extract request attributes/content (no result yet)
        self._apply_extracted_attrs_to_invocation(
            invocation,
            instance,
            normalized_kwargs,
            operation_name,
            result=None,
            extract_attributes_func=extract_attributes_func,
            is_memory_client=is_memory_client,
        )

        self.telemetry_handler.start_memory(invocation)
        hook_context: HookContext = {}
        span = get_current_span()
        safe_call_hook(
            self._memory_before_hook,
            span,
            operation_name,
            instance,
            args,
            dict(kwargs),
            hook_context,
        )
        try:
            result = await func(*args, **kwargs)
            # Post-extract result attributes/content (must happen before stop_memory)
            self._apply_extracted_attrs_to_invocation(
                invocation,
                instance,
                normalized_kwargs,
                operation_name,
                result=result,
                extract_attributes_func=extract_attributes_func,
                is_memory_client=is_memory_client,
            )
            safe_call_hook(
                self._memory_after_hook,
                span,
                operation_name,
                instance,
                args,
                dict(kwargs),
                hook_context,
                result,
                None,
            )
            self.telemetry_handler.stop_memory(invocation)
            return result
        except Exception as e:
            safe_call_hook(
                self._memory_after_hook,
                span,
                operation_name,
                instance,
                args,
                dict(kwargs),
                hook_context,
                None,
                e,
            )
            self.telemetry_handler.fail_memory(
                invocation, Error(message=str(e), type=type(e))
            )
            raise


class VectorStoreWrapper:
    """Vector store subphase wrapper."""

    def __init__(
        self,
        tracer: Tracer,
        *,
        inner_before_hook: InnerBeforeHook = None,
        inner_after_hook: InnerAfterHook = None,
    ):
        """
        Initialize wrapper.

        Args:
            tracer: OpenTelemetry Tracer
        """
        self.tracer = tracer
        self.extractor = VectorOperationAttributeExtractor()
        self._inner_before_hook = inner_before_hook
        self._inner_after_hook = inner_after_hook

    def wrap_vector_operation(self, method_name: str) -> Callable:
        """
        Wrap VectorStore operation method.

        Args:
            method_name: Method name (e.g. 'search', 'insert')

        Returns:
            Wrapper function compatible with wrap_function_wrapper format
        """

        def wrapper(
            wrapped: Callable, instance: Any, args: tuple, kwargs: dict
        ):
            # Check if internal phases enabled
            if not is_internal_phases_enabled():
                return wrapped(*args, **kwargs)

            # Skip Mem0 internal telemetry vector_store to avoid mem0migrations internal spans
            if getattr(instance, "collection_name", None) == "mem0migrations":
                return wrapped(*args, **kwargs)

            # Get span name
            span_name = self._get_span_name(method_name)

            with self.tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
            ) as span:
                result = None
                hook_context: HookContext = {}
                safe_call_hook(
                    self._inner_before_hook,
                    span,
                    "vector",
                    method_name,
                    instance,
                    args,
                    dict(kwargs),
                    hook_context,
                )

                # Store extracted attributes (defined outside try for finally access)
                span_attrs = {}

                try:
                    # Execute original method
                    result = wrapped(*args, **kwargs)

                    # Extract attributes
                    span_attrs = self.extractor.extract_vector_attributes(
                        instance, method_name, kwargs, result
                    )
                    for key, value in span_attrs.items():
                        span.set_attribute(key, value)

                    span.set_status(Status(StatusCode.OK))
                    safe_call_hook(
                        self._inner_after_hook,
                        span,
                        "vector",
                        method_name,
                        instance,
                        args,
                        dict(kwargs),
                        hook_context,
                        result,
                        None,
                    )
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    span.set_attribute(
                        SemanticAttributes.ERROR_TYPE, get_exception_type(e)
                    )
                    safe_call_hook(
                        self._inner_after_hook,
                        span,
                        "vector",
                        method_name,
                        instance,
                        args,
                        dict(kwargs),
                        hook_context,
                        None,
                        e,
                    )
                    raise

        return wrapper

    def _get_span_name(self, method_name: str) -> str:
        """Get span name in format: vector {method_name}"""
        return SpanName.get_subphase_span_name("vector", method_name)


class GraphStoreWrapper:
    """Graph store subphase wrapper."""

    def __init__(
        self,
        tracer: Tracer,
        *,
        inner_before_hook: InnerBeforeHook = None,
        inner_after_hook: InnerAfterHook = None,
    ):
        """
        Initialize wrapper.

        Args:
            tracer: OpenTelemetry Tracer
        """
        self.tracer = tracer
        self.extractor = GraphOperationAttributeExtractor()
        self._inner_before_hook = inner_before_hook
        self._inner_after_hook = inner_after_hook

    def wrap_graph_operation(self, method_name: str) -> Callable:
        """
        Wrap GraphStore operation method.

        Args:
            method_name: Method name (e.g. 'add', 'search')

        Returns:
            Wrapper function compatible with wrap_function_wrapper format
        """

        def wrapper(
            wrapped: Callable, instance: Any, args: tuple, kwargs: dict
        ):
            # Check if internal phases enabled
            if not is_internal_phases_enabled():
                return wrapped(*args, **kwargs)

            # Get span name
            span_name = self._get_span_name(method_name)

            with self.tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
            ) as span:
                result = None
                hook_context: HookContext = {}
                safe_call_hook(
                    self._inner_before_hook,
                    span,
                    "graph",
                    method_name,
                    instance,
                    args,
                    dict(kwargs),
                    hook_context,
                )

                # Store extracted attributes (defined outside try for finally access)
                span_attrs = {}

                try:
                    # Execute original method
                    result = wrapped(*args, **kwargs)

                    # Extract attributes
                    span_attrs = self.extractor.extract_graph_attributes(
                        instance, method_name, result
                    )
                    for key, value in span_attrs.items():
                        span.set_attribute(key, value)

                    span.set_status(Status(StatusCode.OK))
                    safe_call_hook(
                        self._inner_after_hook,
                        span,
                        "graph",
                        method_name,
                        instance,
                        args,
                        dict(kwargs),
                        hook_context,
                        result,
                        None,
                    )
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    span.set_attribute(
                        SemanticAttributes.ERROR_TYPE, get_exception_type(e)
                    )
                    safe_call_hook(
                        self._inner_after_hook,
                        span,
                        "graph",
                        method_name,
                        instance,
                        args,
                        dict(kwargs),
                        hook_context,
                        None,
                        e,
                    )
                    raise

        return wrapper

    def _get_span_name(self, method_name: str) -> str:
        """Get span name in format: graph {method_name}"""
        return SpanName.get_subphase_span_name("graph", method_name)


class RerankerWrapper:
    """Reranker subphase wrapper."""

    def __init__(
        self,
        tracer: Tracer,
        *,
        inner_before_hook: InnerBeforeHook = None,
        inner_after_hook: InnerAfterHook = None,
    ):
        """
        Initialize wrapper.

        Args:
            tracer: OpenTelemetry Tracer
        """
        self.tracer = tracer
        self.extractor = RerankerAttributeExtractor()
        self._inner_before_hook = inner_before_hook
        self._inner_after_hook = inner_after_hook

    def wrap_rerank(self) -> Callable:
        """
        Wrap Reranker.rerank method.

        Returns:
            Wrapper function compatible with wrap_function_wrapper format
        """

        def wrapper(
            wrapped: Callable, instance: Any, args: tuple, kwargs: dict
        ):
            # Check if internal phases enabled
            if not is_internal_phases_enabled():
                return wrapped(*args, **kwargs)

            # Map positional arguments to named parameters for attribute extraction
            # Expected signature: rerank(query, documents, top_k=None, **kwargs)
            derived_kwargs = dict(kwargs)
            if len(args) > 0 and "query" not in derived_kwargs:
                derived_kwargs["query"] = args[0]
            if len(args) > 1 and "documents" not in derived_kwargs:
                derived_kwargs["documents"] = args[1]
            if len(args) > 2 and "top_k" not in derived_kwargs:
                derived_kwargs["top_k"] = args[2]

            with self.tracer.start_as_current_span(
                SpanName.get_subphase_span_name("reranker", "rerank"),
                kind=SpanKind.CLIENT,
            ) as span:
                hook_context: HookContext = {}
                safe_call_hook(
                    self._inner_before_hook,
                    span,
                    "rerank",
                    "rerank",
                    instance,
                    args,
                    dict(kwargs),
                    hook_context,
                )
                # Store extracted attributes (defined outside try for finally access)
                span_attrs = {}

                try:
                    # Extract attributes
                    span_attrs = self.extractor.extract_reranker_attributes(
                        instance, derived_kwargs
                    )
                    for key, value in span_attrs.items():
                        span.set_attribute(key, value)

                    # Execute original method
                    result = wrapped(*args, **kwargs)

                    span.set_status(Status(StatusCode.OK))
                    safe_call_hook(
                        self._inner_after_hook,
                        span,
                        "rerank",
                        "rerank",
                        instance,
                        args,
                        dict(kwargs),
                        hook_context,
                        result,
                        None,
                    )
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    span.set_attribute(
                        SemanticAttributes.ERROR_TYPE, get_exception_type(e)
                    )
                    safe_call_hook(
                        self._inner_after_hook,
                        span,
                        "rerank",
                        "rerank",
                        instance,
                        args,
                        dict(kwargs),
                        hook_context,
                        None,
                        e,
                    )
                    raise

        return wrapper
