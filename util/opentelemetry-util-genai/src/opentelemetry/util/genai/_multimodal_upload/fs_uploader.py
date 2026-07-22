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

"""FsUploader - fsspec-based generic file uploader

Supports fsspec protocols (local, OSS, SLS, etc.).
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import weakref
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple, cast

import fsspec
import httpx

from opentelemetry.instrumentation.utils import suppress_http_instrumentation

# LoongSuite Extension: For Python 3.8 Compatibility
from opentelemetry.util.genai import compatible_hashlib as hashlib
from opentelemetry.util.genai._multimodal_upload._base import (
    Uploader,
    UploadItem,
)
from opentelemetry.util.genai.extended_environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH,
)

_logger = logging.getLogger(__name__)


def hash_content(content: bytes | str) -> str:
    """Return sha256 hex digest for given content.

    If content is str, it is encoded with UTF-8.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content, usedforsecurity=False).hexdigest()


def fs_uploader_hook(
    snapshot: Optional[Any] = None,
) -> Optional[Uploader]:
    """Create default FsUploader from runtime snapshot."""
    if snapshot is None:
        from opentelemetry.util.genai._multimodal_upload.config import (  # pylint: disable=import-outside-toplevel,no-name-in-module
            get_multimodal_config_snapshot,
        )

        snapshot = get_multimodal_config_snapshot()

    base_path = snapshot.effective_storage_base_path or snapshot.storage_base_path
    if not base_path:
        _logger.warning(
            "%s is required but not set, multimodal uploader disabled",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH,
        )
        return None
    return FsUploader(base_path=base_path)


@dataclass
class _Task:
    path: str
    content: Optional[
        bytes
    ]  # Made optional, None for DOWNLOAD_AND_UPLOAD type
    skip_if_exists: bool
    meta: Optional[dict[str, str]]
    content_type: Optional[str]
    source_uri: Optional[str] = None  # Source URI for DOWNLOAD_AND_UPLOAD type
    expected_size: int = 0  # Estimated size for queue management


class FsUploader(Uploader):
    """An fsspec-based generic file uploader for multimodal data

    This class handles actual file upload operations for upload items derived from
    :class:`~opentelemetry.util.genai._multimodal_upload.PreUploader`

    Supports multiple storage backends via fsspec protocols:
    - Local filesystem (file://)
    - Alibaba Cloud OSS (oss://)
    - Alibaba Cloud SLS (sls://)
    - Other fsspec-compatible backends

    Both the ``fsspec`` and ``httpx`` packages should be installed for full functionality.
    For SSL verification control, set :envvar:`OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY`
    to ``false`` to disable SSL verification (default is ``true``).

    Features:
    - Enqueue via upload(path, content, skip_if_exists=True)
    - Background thread pool writes to fsspec filesystem
    - LRU cache avoids re-upload when filename already derived from content hash
    - Supports download-and-upload mode for remote URIs
    - Automatic retry on upload failure

    Args:
        base_path: Complete base path including protocol (e.g., 'oss://bucket', 'sls://project/logstore', 'file:///path')
        max_workers: Maximum number of concurrent upload workers (default: 4)
        max_queue_size: Maximum number of tasks in upload queue (default: 1024)
        max_queue_bytes: Maximum total bytes in queue, 0 for unlimited (default: 0)
        lru_cache_max_size: Maximum size of LRU cache for uploaded files (default: 2048)
        auto_mkdirs: Automatically create parent directories (default: True)
        content_type: Default content type for uploaded files (default: None)
        storage_options: Additional options passed to fsspec (e.g., credentials) (default: None)
        max_upload_retries: Maximum retry attempts for failed uploads, 0 for infinite (default: 10)
        upload_retry_delay: Delay in seconds between retries (default: 1.0)
    """

    def __init__(
        self,
        base_path: str,
        *,
        max_workers: int = 4,
        max_queue_size: int = 1024,
        max_queue_bytes: int = 0,
        lru_cache_max_size: int = 2048,
        auto_mkdirs: bool = True,
        content_type: Optional[str] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        max_upload_retries: int = 10,
        upload_retry_delay: float = 1.0,
    ) -> None:
        # allow passing credentials/endpoint to fsspec
        fs, fs_base = cast(
            Tuple[Any, Any],
            fsspec.url_to_fs(base_path, **(storage_options or {})),
        )
        self._fs = fs
        self._base_path = self._fs.unstrip_protocol(fs_base)
        self._raw_base_path = base_path
        # Protocol parsing: prefer to parse from base_path first, then fall back to fsspec's protocol
        if "://" in base_path:
            self._protocol = base_path.split("://", 1)[0].lower()
        else:
            # Fallback: unknown protocol for local paths
            self._protocol = ""
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._queue_capacity = max_queue_size
        self._max_queue_bytes = max_queue_bytes  # 0 means unlimited
        self._current_queue_bytes = 0
        self._queue_count = 0
        self._queue: Deque[_Task] = deque()
        self._lock = threading.Lock()
        self._queue_cond = threading.Condition(self._lock)  # for shutdown wait
        self._shutdown_event = threading.Event()
        self._lru_uploaded: OrderedDict[str, bool] = OrderedDict()
        self._lru_lock = (
            threading.Lock()
        )  # Dedicated lock to protect LRU cache
        self._lru_capacity = lru_cache_max_size
        self._auto_mkdirs = auto_mkdirs
        self._content_type = content_type
        self._storage_options = storage_options or {}
        self._max_upload_retries = (
            max_upload_retries  # 0 means infinite retries
        )
        self._upload_retry_delay = upload_retry_delay
        self._max_workers = max_workers  # Save for rebuild after fork
        self._shutdown_called = False  # Idempotent flag
        self._ssl_verify = os.environ.get(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY, "true"
        ).lower() not in ("false", "0", "no")

        # background dispatcher
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            name="FsUploader-Dispatcher",
            daemon=True,
        )
        self._dispatcher_thread.start()

        # register_at_fork: Reset state in child process
        # Use weak reference to avoid preventing instance from being GC'd
        if hasattr(os, "register_at_fork"):
            weak_reinit = weakref.WeakMethod(self._at_fork_reinit)
            os.register_at_fork(
                after_in_child=lambda: (ref := weak_reinit()) and ref and ref()
            )
        self._pid = os.getpid()

    @property
    def base_path(self) -> str:
        """Return the normalized base path used by this uploader.

        This can be a local directory or a fully-qualified URL like oss://bucket.
        """
        return self._base_path

    def upload(
        self,
        item: UploadItem,
        *,
        skip_if_exists: bool = True,
    ) -> bool:
        """Enqueue a file upload.

        Args:
            item: Upload task item containing url, data/source_uri, content_type, meta, etc.
            skip_if_exists: Skip if file already exists

        Returns False if the queue is full or uploader is shutting down.
        """
        if self._shutdown_event.is_set():
            return False

        # Validate parameters
        if item.data is None and item.source_uri is None:
            _logger.error(
                "Either data or source_uri must be provided in UploadItem"
            )
            return False

        data = item.data
        if isinstance(data, str):
            data = data.encode("utf-8")

        full_path = self._join(item.url)
        # Best-effort fast path with LRU cache
        if skip_if_exists and self._uploaded_cached(full_path):
            return True

        # Use actual size or estimated size
        content_size = len(data) if data else item.expected_size

        with self._lock:
            # Check queue size limit
            if self._queue_count >= self._queue_capacity:
                _logger.warning("upload queue full, dropping: %s", full_path)
                return False
            # Check bytes limit
            if self._max_queue_bytes > 0 and content_size > 0:
                if (
                    self._current_queue_bytes + content_size
                    > self._max_queue_bytes
                ):
                    _logger.warning(
                        "upload queue bytes limit exceeded (current=%d, incoming=%d, max=%d), dropping: %s",
                        self._current_queue_bytes,
                        content_size,
                        self._max_queue_bytes,
                        full_path,
                    )
                    return False
            self._queue_count += 1
            self._current_queue_bytes += content_size
            self._queue.append(
                _Task(
                    full_path,
                    data,
                    skip_if_exists,
                    item.meta,
                    item.content_type,
                    item.source_uri,
                    item.expected_size,
                )
            )
        return True

    def shutdown(self, timeout: float = 10.0) -> None:
        """
        Gracefully shutdown the uploader.

        Design principles:
        1. Idempotent design: can be called multiple times
        2. Wait for queue to clear first
        3. Set shutdown flag
        4. Wait for running tasks to complete
        5. Shutdown thread pool
        """
        # Idempotent check: return if already shut down
        if self._shutdown_called:
            return
        self._shutdown_called = True

        # If shutdown_event already set (exceptional case), return directly
        if self._shutdown_event.is_set():
            _logger.warning("Uploader already shutdown")
            return

        deadline = time.time() + timeout

        # Phase 1: Wait for queue to clear (limited time)
        with self._queue_cond:
            while self._queue_count > 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    _logger.warning(
                        "shutdown timeout, %d tasks remaining in queue",
                        self._queue_count,
                    )
                    break
                self._queue_cond.wait(timeout=remaining)

        # Phase 2: Set shutdown flag, stop dispatcher
        self._shutdown_event.set()

        # Phase 3: Wait for dispatcher thread to exit
        remaining = max(0.0, deadline - time.time())
        self._dispatcher_thread.join(timeout=remaining)

        # Phase 4: Shutdown thread pool
        # Use wait=False, exit directly after timeout
        # Design principle: use _queue_count as core wait condition
        # - Normal case: Phase 1 waits for _queue_count == 0, exit after all tasks complete
        # - Timeout case: exit directly after timeout, daemon threads will terminate on process exit
        # This ensures minimal data loss while guaranteeing exit within limited time
        self._executor.shutdown(wait=False)

    def _at_fork_reinit(self) -> None:
        """Rebuild resources in child process after fork"""
        _logger.debug("[_at_fork_reinit] FsUploader reinitializing after fork")
        self._lock = threading.Lock()
        self._queue_cond = threading.Condition(self._lock)
        self._lru_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._shutdown_called = False  # Reset idempotent flag
        self._queue.clear()
        self._queue_count = 0
        self._current_queue_bytes = 0
        self._lru_uploaded.clear()

        # Rebuild thread pool
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        # Restart dispatcher
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            name="FsUploader-Dispatcher",
            daemon=True,
        )
        self._dispatcher_thread.start()
        self._pid = os.getpid()

    def _dispatcher_loop(self) -> None:
        while not self._shutdown_event.is_set():
            task: Optional[_Task] = None
            with self._lock:
                if self._queue:
                    task = self._queue.popleft()
            if task is None:
                # No tasks; small sleep without busy loop
                self._shutdown_event.wait(0.01)
                continue
            try:
                self._executor.submit(self._do_upload, task)
            except RuntimeError:
                # executor might be shutting down
                self._release_task(task)

    def _do_upload(self, task: _Task) -> None:
        attempt = 0
        max_retries = self._max_upload_retries
        retry_delay = self._upload_retry_delay

        try:
            # Check cache at the very beginning to avoid unnecessary download and upload
            if task.skip_if_exists and self._file_exists_or_cached(task.path):
                return

            # If this is a download-upload task, download first
            if task.source_uri and task.content is None:
                content = self._download_content(
                    task.source_uri, max_size=30 * 1024 * 1024
                )
                if content is None:
                    _logger.warning(
                        "Failed to download, skip: %s", task.source_uri
                    )
                    return
                task.content = content

                # Update queue bytes (actual size - estimated size)
                size_diff = len(content) - task.expected_size
                if size_diff != 0:
                    with self._lock:
                        self._current_queue_bytes += size_diff

            # ensure dir
            if self._auto_mkdirs:
                self._ensure_parent(task.path)

            # Ensure content exists
            if task.content is None:
                _logger.warning("No content for task: %s", task.path)
                return

            while True:
                attempt += 1
                try:
                    meta_embedded = self._write_file_with_optional_headers(
                        task.path,
                        task.content,
                        task.content_type or self._content_type,
                        task.meta,
                    )

                    # Sidecar .meta JSON for non-OSS or when headers not supported
                    if task.meta and not meta_embedded:
                        self._write_sidecar_meta(task.path, task.meta)

                    # mark cache
                    self._mark_uploaded(task.path)
                    return  # success
                except (OSError, IOError, RuntimeError) as exc:
                    # OSError/IOError: File system error (network storage, local disk)
                    # RuntimeError: fsspec underlying error (e.g., OSS/SLS auth failure)
                    # Check if we should retry (max_retries=0 means infinite)
                    if 0 < max_retries <= attempt:
                        _logger.exception(
                            "upload failed after %d attempts: %s",
                            attempt,
                            task.path,
                        )
                        return
                    _logger.warning(
                        "upload attempt %d failed for %s: %s, retrying in %.1fs...",
                        attempt,
                        task.path,
                        str(exc),
                        retry_delay,
                    )
                    time.sleep(retry_delay)
        finally:
            self._release_task(task)

    def _release_task(self, task: _Task) -> None:
        """Release task resources and notify shutdown waiter."""
        with self._queue_cond:
            self._queue_count -= 1
            # Use actual size or estimated size
            size_to_release = (
                len(task.content) if task.content else task.expected_size
            )
            self._current_queue_bytes -= size_to_release
            self._queue_cond.notify_all()

    def _download_content(
        self, uri: str, max_size: int, timeout: float = 30.0
    ) -> Optional[bytes]:
        """Download URI content using stream + BytesIO to avoid memory doubling"""
        # Use suppress_http_instrumentation to avoid internal HTTP requests being captured by probe
        with suppress_http_instrumentation():
            try:
                with httpx.Client(
                    timeout=timeout, verify=self._ssl_verify
                ) as client:
                    with client.stream("GET", uri) as response:
                        # Explicitly reject 3xx redirects, to prevent old httpx versions from silently getting incorrect body
                        if 300 <= response.status_code < 400:
                            raise httpx.HTTPStatusError(
                                "Redirect not allowed",
                                request=response.request,
                                response=response,
                            )
                        response.raise_for_status()
                        buffer = io.BytesIO()
                        try:
                            for chunk in response.iter_bytes(
                                chunk_size=64 * 1024
                            ):
                                if buffer.tell() + len(chunk) > max_size:
                                    _logger.warning(
                                        "Download exceeds max size %d, abort: %s",
                                        max_size,
                                        uri,
                                    )
                                    return None
                                buffer.write(chunk)
                            return buffer.getvalue()
                        finally:
                            buffer.close()
            except (httpx.HTTPError, OSError, ValueError) as exc:
                # httpx.HTTPError: HTTP request failure (timeout, connection error, HTTP error)
                # OSError: Low-level network/system error
                # ValueError: Data parsing error
                _logger.warning("Failed to download: %s, error: %s", uri, exc)
                return None

    def _join(self, path: str) -> str:
        # If caller passes a fully-qualified URL (e.g. oss://bucket/key), keep it as-is
        if "://" in path:
            return path
        if path.startswith("/"):
            return path
        return os.path.join(self._base_path, path)

    def _ensure_parent(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent and not self._fs.exists(parent):
            try:
                # Attempt to call makedirs if available on filesystem implementation
                makedirs = getattr(self._fs, "makedirs", None)
                if callable(makedirs):
                    makedirs(parent, exist_ok=True)
            except (OSError, RuntimeError, AttributeError):
                # OSError: File system error
                # RuntimeError: fsspec underlying error
                # AttributeError: _fs doesn't have makedirs method (some fsspec implementations)
                # Best effort; race-safe for remote fs
                pass

    def _file_exists_or_cached(self, path: str) -> bool:
        if self._uploaded_cached(path):
            return True
        exists = self._fs.exists(path)
        if exists:
            self._mark_uploaded(path)
        return exists

    def _uploaded_cached(self, path: str) -> bool:
        with self._lru_lock:
            if path in self._lru_uploaded:
                self._lru_uploaded.move_to_end(path)
                return True
            return False

    def _mark_uploaded(self, path: str) -> None:
        with self._lru_lock:
            self._lru_uploaded[path] = True
            if len(self._lru_uploaded) > self._lru_capacity:
                self._lru_uploaded.popitem(last=False)

    # --- meta handling helpers ---

    def _is_oss(self) -> bool:
        return self._protocol == "oss"

    @staticmethod
    def _build_oss_headers(meta: dict[str, str]) -> dict[str, str]:
        """Build OSS metadata headers"""
        headers: Dict[str, str] = {}
        for key, value in meta.items():
            headers[f"x-oss-meta-{key}"] = str(value)
        return headers

    @staticmethod
    def _build_sls_meta(meta: dict[str, str]) -> dict[str, str]:
        """Build SLS metadata"""
        sls_meta: Dict[str, str] = {}
        for key, value in (meta or {}).items():
            sls_meta[f"x-log-meta-{key}"] = str(value)
        return sls_meta

    def _is_sls(self) -> bool:
        return self._protocol == "sls"

    def _write_file_with_optional_headers(
        self,
        path: str,
        content: bytes,
        content_type: Optional[str],
        meta: Optional[dict[str, str]],
    ) -> bool:
        """Write file; on OSS/SLS use pipe_file with headers, else fallback to open.

        Returns True if metadata/content_type was embedded into the object.
        """
        # Prefer native OSS path so headers (Content-Type, x-oss-meta-*) are honored
        if self._is_oss():
            headers: Dict[str, Any] = {}
            if content_type:
                headers["Content-Type"] = content_type
            if meta:
                headers.update(self._build_oss_headers(meta))
            try:
                # pipe_file delegates to put_object(..., headers=...)
                self._fs.pipe_file(path, content, headers=headers)
                return bool(headers)
            except (OSError, RuntimeError, AttributeError):
                # OSError: File system error
                # RuntimeError: OSS SDK error
                # AttributeError: _fs doesn't have pipe_file method
                _logger.exception(
                    "OSS pipe_file failed, falling back to standard write: %s",
                    path,
                )
                # fall through to generic write

        # Prefer native SLS path so headers (Content-Type, x-log-meta-*) are honored
        if self._is_sls():
            sls_headers: Dict[str, Any] = {}
            if content_type:
                sls_headers["Content-Type"] = content_type
            if meta:
                sls_headers.update(self._build_sls_meta(meta))
            try:
                # pipe_file delegates to put_object(..., headers=...)
                self._fs.pipe_file(path, content, headers=sls_headers)
                return bool(sls_headers)
            except (OSError, RuntimeError, AttributeError):
                # OSError: File system error
                # RuntimeError: SLS SDK error
                # AttributeError: _fs doesn't have pipe_file method
                _logger.exception(
                    "SLS pipe_file failed, falling back to standard write: %s",
                    path,
                )
                # fall through to generic write

        # Generic fsspec write; some backends accept content_type
        base_kwargs: Dict[str, Any] = {}
        if content_type:
            base_kwargs["content_type"] = content_type
        try:
            with self._fs.open(path, "wb", **base_kwargs) as file_obj:
                file_obj.write(content)
        except TypeError:
            with self._fs.open(path, "wb") as file_obj:
                file_obj.write(content)

        return False

    def _write_sidecar_meta(self, path: str, meta: dict[str, str]) -> None:
        sidecar = f"{path}.meta"
        payload_meta = self._build_sls_meta(meta) if self._is_sls() else meta
        data = json.dumps(payload_meta, ensure_ascii=False, sort_keys=True)
        try:
            with self._fs.open(
                sidecar, "w", content_type="application/json; charset=utf-8"
            ) as file_obj:
                file_obj.write(data)
        except TypeError:
            with self._fs.open(sidecar, "w") as file_obj:
                file_obj.write(data)
