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
Multimodal Processing Mixin for GenAI telemetry handlers.

This module provides the `MultimodalProcessingMixin` class which adds async
multimodal processing capabilities to telemetry handlers. It handles:
- Async queue-based multimodal data processing
- Non-blocking upload of multimodal content (images, audio, etc.)
- Graceful degradation when queue is full

Usage:
    class MyHandler(MultimodalProcessingMixin, TelemetryHandler):
        def __init__(self, ...):
            super().__init__(...)
            self._init_multimodal()
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from time import time_ns
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

from opentelemetry._logs import Logger as OtelLogger
from opentelemetry.trace import Span
from opentelemetry.util.genai.extended_semconv import (
    gen_ai_extended_attributes as GenAIEx,
)
from opentelemetry.util.genai.extended_span_utils import (
    _apply_invoke_agent_finish_attributes,
    _maybe_emit_invoke_agent_event,
)
from opentelemetry.util.genai.extended_types import InvokeAgentInvocation
from opentelemetry.util.genai.handler import _safe_detach
from opentelemetry.util.genai.span_utils import (
    _apply_error_attributes,
    _apply_llm_finish_attributes,
    _maybe_emit_llm_event,
)
from opentelemetry.util.genai.types import (
    Base64Blob,
    Blob,
    ContentCapturingMode,
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Uri,
)
from opentelemetry.util.genai.utils import (  # pylint: disable=no-name-in-module
    gen_ai_json_dumps,
    get_content_capturing_mode,
    get_multimodal_upload_mode,
    is_experimental_mode,
)

if TYPE_CHECKING:
    from opentelemetry.util.genai._multimodal_upload._base import (
        PreUploader,
        Uploader,
    )

_logger = logging.getLogger(__name__)

# Async queue maximum length
_MAX_ASYNC_QUEUE_SIZE = 1000

# Invocation types that carry multimodal messages
_MultimodalInvocation = Union[LLMInvocation, InvokeAgentInvocation]

# Task method literals
_TaskMethod = Literal["stop_llm", "fail_llm", "stop_agent", "fail_agent"]


@dataclass
class _MultimodalAsyncTask:
    """Async multimodal processing task"""

    invocation: _MultimodalInvocation
    method: _TaskMethod
    error: Optional[Error] = None
    handler: Optional[Any] = None  # TelemetryHandler instance


class MultimodalProcessingMixin:
    """
    Mixin class that adds async multimodal processing capabilities.

    This mixin should be used with TelemetryHandler or its subclasses.
    It provides non-blocking multimodal data processing to avoid blocking
    user applications during upload operations.

    Class-level resources (queue, worker thread) are shared across all
    handler instances in the process.
    """

    # Class-level shared async processing resources
    _async_queue: ClassVar[
        Optional["queue.Queue[Optional[_MultimodalAsyncTask]]"]
    ] = None
    _async_worker: ClassVar[Optional[threading.Thread]] = None
    _async_lock: ClassVar[threading.Lock] = threading.Lock()
    _atexit_handler: ClassVar[Optional[object]] = None
    _shutdown_atexit_lock: ClassVar[threading.Lock] = threading.Lock()
    _shutdown_lock: ClassVar[threading.Lock] = threading.Lock()
    _shutdown_called: ClassVar[bool] = False

    # Instance-level attributes (initialized by _init_multimodal)
    _multimodal_enabled: bool

    def _init_multimodal(self) -> None:
        """Initialize multimodal-related instance attributes, called in subclass __init__"""
        self._multimodal_enabled = False

        if get_multimodal_upload_mode() == "none":
            return

        try:
            capture_enabled = (
                is_experimental_mode()
                and get_content_capturing_mode()
                in (
                    ContentCapturingMode.SPAN_ONLY,
                    ContentCapturingMode.SPAN_AND_EVENT,
                )
            )
        except ValueError:
            # get_content_capturing_mode raises ValueError when GEN_AI stability mode is DEFAULT
            capture_enabled = False

        if not capture_enabled:
            return

        uploader, pre_uploader = self._get_uploader_and_pre_uploader()
        if uploader is not None and pre_uploader is not None:
            self._multimodal_enabled = True

    # ==================== Public Methods ====================

    def process_multimodal_stop(
        self,
        invocation: _MultimodalInvocation,
        method: _TaskMethod,
    ) -> bool:
        """Process multimodal stop request

        Args:
            invocation: LLM or Agent invocation object
            method: Task method for dispatch ("stop_llm" or "stop_agent")

        Returns:
            bool: Whether handled (True = async processed, False = no multimodal)
        """
        if invocation.context_token is None or invocation.span is None:
            return False

        if not self._should_async_process(invocation):
            return False

        # 1. Detach context immediately (let user code continue)
        _safe_detach(invocation.context_token)

        # 2. Ensure worker is started
        self._ensure_async_worker()

        # 3. Try to put into queue (non-blocking)
        async_queue = self.__class__._async_queue
        if async_queue is None:
            self._fallback_stop(invocation, method)
            return True
        try:
            async_queue.put_nowait(
                _MultimodalAsyncTask(
                    invocation=invocation,
                    method=method,
                    handler=self,
                )
            )
        except queue.Full:
            _logger.warning(
                "Multimodal queue full, skipping multimodal processing"
            )
            self._fallback_stop(invocation, method)

        return True

    def process_multimodal_fail(
        self,
        invocation: _MultimodalInvocation,
        error: Error,
        method: _TaskMethod,
    ) -> bool:
        """Process multimodal fail request

        Args:
            invocation: LLM or Agent invocation object
            error: Error information
            method: Task method for dispatch ("fail_llm" or "fail_agent")

        Returns:
            bool: Whether handled
        """
        if invocation.context_token is None or invocation.span is None:
            return False

        if not self._should_async_process(invocation):
            return False

        _safe_detach(invocation.context_token)
        self._ensure_async_worker()

        async_queue = self.__class__._async_queue
        if async_queue is None:
            self._fallback_fail(invocation, error, method)
            return True
        try:
            async_queue.put_nowait(
                _MultimodalAsyncTask(
                    invocation=invocation,
                    method=method,
                    error=error,
                    handler=self,
                )
            )
        except queue.Full:
            _logger.warning(
                "Multimodal queue full, skipping multimodal processing"
            )
            self._fallback_fail(invocation, error, method)

        return True

    @classmethod
    def shutdown_multimodal_worker(cls, timeout: float = 5.0) -> None:
        """Gracefully shutdown async worker

        Called by shutdown during graceful exit.

        Strategy:
        1. Try to send None signal to queue within timeout
        2. If send succeeds, wait remaining time for worker to process queue tasks
        3. If send times out, worker may have unprocessed data (but won't block caller)
        """
        if cls._async_worker is None or not cls._async_worker.is_alive():
            return

        if cls._async_queue is None:
            return

        start_time = time.monotonic()
        try:
            # Try to send shutdown signal within timeout
            cls._async_queue.put(None, timeout=timeout)
        except queue.Full:
            # Queue full, timed out, cannot send signal
            _logger.warning(
                "Async worker shutdown: queue full, timeout waiting to send signal"
            )
            return

        # Calculate remaining timeout
        elapsed = time.monotonic() - start_time
        remaining = max(0.0, timeout - elapsed)

        # Wait for worker to complete remaining tasks
        cls._async_worker.join(timeout=remaining)

        # Clean up state
        cls._async_worker = None
        cls._async_queue = None

    @classmethod
    def _ensure_multimodal_shutdown_atexit_registered(cls) -> None:
        """Register a single process-level atexit shutdown callback."""
        if cls._atexit_handler is not None:
            return
        with cls._shutdown_atexit_lock:
            if cls._atexit_handler is not None:
                return
            cls._atexit_handler = atexit.register(cls._shutdown_for_exit)

    @classmethod
    def _shutdown_for_exit(cls) -> None:
        """atexit callback entrypoint for multimodal graceful shutdown."""
        cls.shutdown()

    @classmethod
    def shutdown(
        cls,
        worker_timeout: float = 5.0,
        pre_uploader_timeout: float = 2.0,
        uploader_timeout: float = 5.0,
    ) -> None:
        """Shutdown multimodal worker, pre-uploader and uploader in order."""
        with cls._shutdown_lock:
            if cls._shutdown_called:
                return
            cls._shutdown_called = True

        cls.shutdown_multimodal_worker(worker_timeout)
        cls._shutdown_pre_uploader(pre_uploader_timeout)
        cls._shutdown_uploader(uploader_timeout)

    @classmethod
    def _shutdown_pre_uploader(cls, timeout: float) -> None:
        try:
            from opentelemetry.util.genai._multimodal_upload import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
                get_pre_uploader,
            )

            pre_uploader = get_pre_uploader()
            if pre_uploader is not None and hasattr(pre_uploader, "shutdown"):
                pre_uploader.shutdown(timeout=timeout)
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning("Error shutting down PreUploader: %s", exc)

    @classmethod
    def _shutdown_uploader(cls, timeout: float) -> None:
        try:
            from opentelemetry.util.genai._multimodal_upload import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
                get_uploader,
            )

            uploader = get_uploader()
            if uploader is not None and hasattr(uploader, "shutdown"):
                uploader.shutdown(timeout=timeout)
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning("Error shutting down Uploader: %s", exc)

    @classmethod
    def _at_fork_reinit(cls) -> None:
        """Reset class-level state in child process after fork"""
        _logger.debug(
            "[_at_fork_reinit] MultimodalProcessingMixin reinitializing after fork"
        )
        cls._async_lock = threading.Lock()
        cls._async_queue = None
        cls._async_worker = None
        cls._atexit_handler = None

    # ==================== Internal Methods ====================

    def _should_async_process(self, invocation: _MultimodalInvocation) -> bool:
        """Determine whether async processing is needed

        Condition: Has multimodal data and multimodal upload switch is not 'none'
        """
        if not self._multimodal_enabled:
            return False

        return MultimodalProcessingMixin._quick_has_multimodal(invocation)

    @staticmethod
    def _quick_has_multimodal(invocation: _MultimodalInvocation) -> bool:
        """Quick detection of multimodal data (O(n), no network)"""

        def _check_messages(
            messages: Optional[List[InputMessage] | List[OutputMessage]],
        ) -> bool:
            if not messages:
                return False
            for msg in messages:
                if not hasattr(msg, "parts") or not msg.parts:
                    continue
                for part in msg.parts:
                    if isinstance(part, (Base64Blob, Blob, Uri)):
                        return True
            return False

        return _check_messages(invocation.input_messages) or _check_messages(
            invocation.output_messages
        )

    @classmethod
    def _ensure_async_worker(cls) -> None:
        """Ensure worker thread is started (double-checked locking)"""
        if cls._async_worker is not None and cls._async_worker.is_alive():
            return

        with cls._async_lock:
            if cls._async_worker is not None and cls._async_worker.is_alive():
                return

            if cls._async_queue is None:
                cls._async_queue = queue.Queue(maxsize=_MAX_ASYNC_QUEUE_SIZE)

            cls._async_worker = threading.Thread(
                target=cls._async_worker_loop,
                name="MultimodalProcessingMixin-AsyncWorker",
                daemon=True,
            )
            cls._async_worker.start()

    @classmethod
    def _async_worker_loop(cls) -> None:
        """Worker thread main loop

        Loop until None signal received to exit. Use blocking get() to avoid CPU spinning.
        """
        while True:
            # Save queue reference to prevent being set to None during shutdown
            async_queue = cls._async_queue
            if async_queue is None:
                break

            try:
                task = async_queue.get()  # Blocking wait
            except (EOFError, OSError) as exc:
                # EOFError: Queue closed or broken pipe
                # OSError: Low-level system error in queue operations
                _logger.warning("Queue error in async worker: %s", exc)
                break  # Queue exception, exit loop

            if task is None:  # shutdown signal
                async_queue.task_done()
                break

            handler = task.handler
            if handler is None:
                continue

            try:
                handler._dispatch_task(task)
            except (
                AttributeError,
                TypeError,
                RuntimeError,
                OSError,
                ValueError,
            ) as exc:
                # AttributeError: Handler method or attribute missing
                # TypeError: Method call argument type error
                # RuntimeError: Upload or span operation runtime error
                # OSError: Network or file system error from upload
                # ValueError: Data validation error
                _logger.warning("Multimodal async processing error: %s", exc)
                # Ensure span is ended
                try:
                    end_time_ns = (
                        MultimodalProcessingMixin._compute_end_time_ns(
                            task.invocation
                        )
                    )
                    span = task.invocation.span
                    if span is not None:
                        span.end(end_time=end_time_ns)
                except (
                    AttributeError,
                    TypeError,
                    RuntimeError,
                ) as cleanup_exc:
                    # AttributeError: Span object missing end method
                    # TypeError: end_time parameter type error
                    # RuntimeError: Span already ended or invalid state
                    _logger.debug("Failed to cleanup span: %s", cleanup_exc)
            finally:
                # Use local variable to avoid race condition
                async_queue.task_done()

    def _dispatch_task(self, task: _MultimodalAsyncTask) -> None:
        """Dispatch task to the appropriate handler method based on task.method"""
        if task.method == "stop_llm":
            self._async_stop_llm(task)
        elif task.method == "fail_llm":
            self._async_fail_llm(task)
        elif task.method == "stop_agent":
            self._async_stop_invoke_agent(task)
        elif task.method == "fail_agent":
            self._async_fail_invoke_agent(task)

    # ==================== LLM Async Methods ====================

    def _async_stop_llm(self, task: _MultimodalAsyncTask) -> None:
        """Async stop LLM invocation (executed in worker thread)"""
        invocation = task.invocation
        span = invocation.span
        if span is None:
            return

        # 1. Get uploader and process multimodal data
        uploader, pre_uploader = self._get_uploader_and_pre_uploader()
        if uploader is not None and pre_uploader is not None:
            self._upload_and_set_metadata(
                span, invocation, uploader, pre_uploader
            )

        # 2. Execute original attribute setting
        _apply_llm_finish_attributes(span, invocation)  # type: ignore[arg-type]

        # 3. Record metrics (using TelemetryHandler's method)
        self._record_llm_metrics(invocation, span)  # type: ignore[attr-defined]

        # 4. Send event
        _maybe_emit_llm_event(self._logger, span, invocation)  # type: ignore[attr-defined]

        # 5. Calculate correct end time and end span
        end_time_ns = MultimodalProcessingMixin._compute_end_time_ns(
            invocation
        )
        span.end(end_time=end_time_ns)

    def _async_fail_llm(self, task: _MultimodalAsyncTask) -> None:
        """Async fail LLM invocation (executed in worker thread)"""
        invocation = task.invocation
        error = task.error
        span = invocation.span
        if span is None or error is None:
            return

        # 1. Get uploader and process multimodal data
        uploader, pre_uploader = self._get_uploader_and_pre_uploader()
        if uploader is not None and pre_uploader is not None:
            self._upload_and_set_metadata(
                span, invocation, uploader, pre_uploader
            )

        # 2. Set attributes
        _apply_llm_finish_attributes(span, invocation)  # type: ignore[arg-type]
        _apply_error_attributes(span, error)

        # 3. Record metrics
        error_type = getattr(error.type, "__qualname__", None)
        self._record_llm_metrics(invocation, span, error_type=error_type)  # type: ignore[attr-defined]

        # 4. Send event
        _maybe_emit_llm_event(self._logger, span, invocation, error)  # type: ignore[attr-defined]

        # 5. End span
        end_time_ns = MultimodalProcessingMixin._compute_end_time_ns(
            invocation
        )
        span.end(end_time=end_time_ns)

    # ==================== Agent Async Methods ====================

    def _async_stop_invoke_agent(self, task: _MultimodalAsyncTask) -> None:
        """Async stop Agent invocation (executed in worker thread)"""
        invocation = task.invocation
        if not isinstance(invocation, InvokeAgentInvocation):
            return
        span = invocation.span
        if span is None:
            return

        # 1. Get uploader and process multimodal data
        uploader, pre_uploader = self._get_uploader_and_pre_uploader()
        if uploader is not None and pre_uploader is not None:
            self._upload_and_set_metadata(
                span, invocation, uploader, pre_uploader
            )

        # 2. Execute attribute setting
        _apply_invoke_agent_finish_attributes(span, invocation)

        # 3. Record metrics
        cast(Any, self)._record_extended_metrics(span, invocation)

        # 4. Send event
        event_logger = cast(
            Optional[OtelLogger], getattr(self, "_logger", None)
        )
        _maybe_emit_invoke_agent_event(
            event_logger,
            span,
            invocation,
        )

        # 5. Calculate correct end time and end span
        end_time_ns = MultimodalProcessingMixin._compute_end_time_ns(
            invocation
        )
        span.end(end_time=end_time_ns)

    def _async_fail_invoke_agent(self, task: _MultimodalAsyncTask) -> None:
        """Async fail Agent invocation (executed in worker thread)"""
        invocation = task.invocation
        if not isinstance(invocation, InvokeAgentInvocation):
            return
        error = task.error
        span = invocation.span
        if span is None or error is None:
            return

        # 1. Get uploader and process multimodal data
        uploader, pre_uploader = self._get_uploader_and_pre_uploader()
        if uploader is not None and pre_uploader is not None:
            self._upload_and_set_metadata(
                span, invocation, uploader, pre_uploader
            )

        # 2. Set attributes
        _apply_invoke_agent_finish_attributes(span, invocation)
        _apply_error_attributes(span, error)

        # 3. Record metrics
        error_type = getattr(error.type, "__qualname__", None)
        cast(Any, self)._record_extended_metrics(
            span, invocation, error_type=error_type
        )

        # 4. Send event
        event_logger = cast(
            Optional[OtelLogger], getattr(self, "_logger", None)
        )
        _maybe_emit_invoke_agent_event(
            event_logger,
            span,
            invocation,
            error,
        )

        # 5. End span
        end_time_ns = MultimodalProcessingMixin._compute_end_time_ns(
            invocation
        )
        span.end(end_time=end_time_ns)

    # ==================== Fallback Methods ====================

    def _fallback_stop(
        self,
        invocation: _MultimodalInvocation,
        method: _TaskMethod,
    ) -> None:
        """Sync degradation for stop: skip multimodal, end span with attributes"""
        span = invocation.span
        if span is None:
            return
        if method == "stop_llm":
            if not isinstance(invocation, LLMInvocation):
                return
            _apply_llm_finish_attributes(span, invocation)
            cast(Any, self)._record_llm_metrics(invocation, span)
            event_logger = cast(
                Optional[OtelLogger], getattr(self, "_logger", None)
            )
            _maybe_emit_llm_event(event_logger, span, invocation)
        elif method == "stop_agent":
            if not isinstance(invocation, InvokeAgentInvocation):
                return
            _apply_invoke_agent_finish_attributes(span, invocation)
            cast(Any, self)._record_extended_metrics(span, invocation)
            event_logger = cast(
                Optional[OtelLogger], getattr(self, "_logger", None)
            )
            _maybe_emit_invoke_agent_event(event_logger, span, invocation)
        end_time_ns = MultimodalProcessingMixin._compute_end_time_ns(
            invocation
        )
        span.end(end_time=end_time_ns)

    def _fallback_fail(
        self,
        invocation: _MultimodalInvocation,
        error: Error,
        method: _TaskMethod,
    ) -> None:
        """Sync degradation for fail: skip multimodal, end span with error"""
        span = invocation.span
        if span is None:
            return
        error_type = getattr(error.type, "__qualname__", None)
        if method == "fail_llm":
            if not isinstance(invocation, LLMInvocation):
                return
            _apply_llm_finish_attributes(span, invocation)
            _apply_error_attributes(span, error)
            cast(Any, self)._record_llm_metrics(
                invocation, span, error_type=error_type
            )
            event_logger = cast(
                Optional[OtelLogger], getattr(self, "_logger", None)
            )
            _maybe_emit_llm_event(event_logger, span, invocation, error)
        elif method == "fail_agent":
            if not isinstance(invocation, InvokeAgentInvocation):
                return
            _apply_invoke_agent_finish_attributes(span, invocation)
            _apply_error_attributes(span, error)
            cast(Any, self)._record_extended_metrics(
                span, invocation, error_type=error_type
            )
            event_logger = cast(
                Optional[OtelLogger], getattr(self, "_logger", None)
            )
            _maybe_emit_invoke_agent_event(
                event_logger, span, invocation, error
            )
        end_time_ns = MultimodalProcessingMixin._compute_end_time_ns(
            invocation
        )
        span.end(end_time=end_time_ns)

    # ==================== Timing Helpers ====================

    @staticmethod
    def _compute_end_time_ns(invocation: _MultimodalInvocation) -> int:
        """Calculate absolute time (nanoseconds) based on monotonic time"""
        if not invocation.monotonic_end_s or not invocation.monotonic_start_s:
            return time_ns()

        # Get start_time from span (already in ns)
        start_time_ns = getattr(invocation.span, "_start_time", None)
        if not start_time_ns:
            return time_ns()

        # Calculate duration (ns)
        duration_ns = int(
            (invocation.monotonic_end_s - invocation.monotonic_start_s) * 1e9
        )
        return start_time_ns + duration_ns

    # ==================== Multimodal Helper Methods ====================

    def _get_uploader_and_pre_uploader(  # pylint: disable=no-self-use
        self,
    ) -> Tuple[Any, Any]:
        """Lazily get uploader and pre_uploader to avoid circular imports

        Note: Keep as instance method for consistency with other methods in this mixin
        """
        try:
            from opentelemetry.util.genai._multimodal_upload import (  # pylint: disable=import-outside-toplevel,no-name-in-module  # noqa: PLC0415
                get_or_load_uploader_pair,
            )

            return get_or_load_uploader_pair()
        except ImportError:
            return None, None

    def _upload_and_set_metadata(
        self,
        span: Span,
        invocation: _MultimodalInvocation,
        uploader: "Uploader",
        pre_uploader: "PreUploader",
    ) -> None:
        """Upload multimodal data and set metadata attributes on span"""
        self._separate_and_upload(span, invocation, uploader, pre_uploader)

        input_metadata, output_metadata = (
            MultimodalProcessingMixin._extract_multimodal_metadata(
                invocation.input_messages, invocation.output_messages
            )
        )
        if input_metadata:
            span.set_attribute(
                GenAIEx.GEN_AI_INPUT_MULTIMODAL_METADATA,
                gen_ai_json_dumps(input_metadata),
            )
        if output_metadata:
            span.set_attribute(
                GenAIEx.GEN_AI_OUTPUT_MULTIMODAL_METADATA,
                gen_ai_json_dumps(output_metadata),
            )

    def _separate_and_upload(  # pylint: disable=no-self-use
        self,
        span: Span,
        invocation: _MultimodalInvocation,
        uploader: "Uploader",
        pre_uploader: "PreUploader",
    ) -> None:
        """Separate multimodal data and submit for upload"""
        try:
            span_context = span.get_span_context()
            start_time_ns = getattr(span, "_start_time", None) or int(
                time.time() * 1_000_000_000
            )

            upload_items = pre_uploader.pre_upload(
                span_context,
                start_time_ns,
                invocation.input_messages,
                invocation.output_messages,
            )

            for item in upload_items:
                uploader.upload(item)
        except (
            AttributeError,
            TypeError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            # AttributeError: Uploader/pre_uploader method missing
            # TypeError: Method argument type error
            # OSError: File system or network error
            # RuntimeError: Upload operation failed
            # ValueError: Data validation error
            _logger.debug("Error in _separate_and_upload: %s", exc)

    @staticmethod
    def _extract_multimodal_metadata(
        input_messages: Optional[List[InputMessage]],
        output_messages: Optional[List[OutputMessage]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract multimodal metadata from messages.

        Important:
        - URI metadata extraction is based on the final message parts.
        - It is independent from download/replace success in pre-uploader.
        - When URI replacement is skipped (e.g. download disabled) or fails,
          the original URI should still remain in messages and be reported here.
        """

        def _extract_from_messages(
            messages: Optional[List[InputMessage] | List[OutputMessage]],
        ) -> List[Dict[str, Any]]:
            metadata: List[Dict[str, Any]] = []
            if not messages:
                return metadata
            for msg in messages:
                if not hasattr(msg, "parts") or not msg.parts:
                    continue
                for part in msg.parts:
                    if isinstance(part, Uri):
                        metadata.append(
                            {
                                "type": "uri",
                                "mime_type": part.mime_type,
                                "uri": part.uri,
                                "modality": part.modality,
                            }
                        )
            return metadata

        return (
            _extract_from_messages(input_messages),
            _extract_from_messages(output_messages),
        )


# Module-level fork handler registration
if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        after_in_child=MultimodalProcessingMixin._at_fork_reinit
    )
