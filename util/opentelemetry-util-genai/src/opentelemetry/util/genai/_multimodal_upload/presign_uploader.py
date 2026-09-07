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

"""Uploader for the pre-authorized OSS mode.

Each object is written with a short-lived URL obtained from the ARMS presign
API, so the agent needs no OSS credentials and no OSS SDK.
"""

from __future__ import annotations

import io
import logging
import os
import random
import threading
import time
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx

from opentelemetry.instrumentation.utils import suppress_http_instrumentation
from opentelemetry.util.genai._multimodal_upload._base import (
    PreUploader,
    Uploader,
    UploadItem,
)
from opentelemetry.util.genai._multimodal_upload.config import (  # pylint: disable=no-name-in-module
    DEFAULT_SLS_LOGSTORE,
    format_sls_base_path,
)
from opentelemetry.util.genai._multimodal_upload.presign_client import (
    MultimodalPresignClient,
    PresignAuthError,
    PresignConfigError,
    PresignError,
    PresignRetryableError,
    resolve_presign_endpoint,
)
from opentelemetry.util.genai._multimodal_upload.usage_recorder import (
    get_multimodal_usage_recorder,
    provider_label_from_protocol,
)
from opentelemetry.util.genai.extended_environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_TIMEOUT,
)

_logger = logging.getLogger(__name__)

_USAGE_METRICS_PROVIDER = provider_label_from_protocol("oss")
_MAX_DOWNLOAD_BYTES = 30 * 1024 * 1024
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_WORKERS = 4
_DEFAULT_MAX_QUEUE_SIZE = 1024
_DEFAULT_MAX_QUEUE_BYTES = 1024 * 1024 * 1024
_DEFAULT_LRU_CACHE_MAX_SIZE = 2048
_DEFAULT_MAX_UPLOAD_ATTEMPTS = 3
_DEFAULT_UPLOAD_RETRY_DELAY = 0.2


class PresignUploadConfigError(PresignConfigError):
    """The presign uploader target path is unusable."""


class _PresignedUploadRejected(Exception):
    """The presigned PUT was rejected by the storage service."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.retryable = retryable


@dataclass
class _Task:
    full_path: str
    object_key: str
    content: Optional[bytes]
    source_uri: Optional[str]
    expected_size: int
    reserved_size: int
    skip_if_exists: bool


def _presign_timeout() -> float:
    raw = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_TIMEOUT, ""
    ).strip()
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        timeout = float(raw)
    except ValueError:
        _logger.warning(
            "Invalid %s=%r, falling back to %.1fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_TIMEOUT,
            raw,
            _DEFAULT_TIMEOUT,
        )
        return _DEFAULT_TIMEOUT
    return timeout if timeout > 0 else _DEFAULT_TIMEOUT


class PresignUploader(Uploader):
    """Bounded asynchronous uploader driven by ARMS presigned URLs."""

    def __init__(
        self,
        base_path: str,
        *,
        client: MultimodalPresignClient,
        timeout: float = _DEFAULT_TIMEOUT,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        max_queue_size: int = _DEFAULT_MAX_QUEUE_SIZE,
        max_queue_bytes: int = _DEFAULT_MAX_QUEUE_BYTES,
        lru_cache_max_size: int = _DEFAULT_LRU_CACHE_MAX_SIZE,
        max_upload_attempts: int = _DEFAULT_MAX_UPLOAD_ATTEMPTS,
        upload_retry_delay: float = _DEFAULT_UPLOAD_RETRY_DELAY,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._project, self._logstore, self._prefix = self._parse_base_path(
            base_path
        )
        self._base_path = (
            format_sls_base_path(self._project, self._logstore, self._prefix)
            or f"sls://{self._project}/{self._logstore}"
        )
        self._client = client
        self._timeout = timeout
        self._ssl_verify = os.environ.get(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY,
            "true",
        ).lower() not in ("false", "0", "no")
        self._owns_http_client = http_client is None
        self._http_client = http_client or self._new_http_client()

        self._max_workers = max_workers
        self._queue_capacity = max_queue_size
        self._max_queue_bytes = max_queue_bytes
        self._lru_capacity = lru_cache_max_size
        self._max_upload_attempts = max(1, max_upload_attempts)
        self._upload_retry_delay = max(0.0, upload_retry_delay)

        self._lock = threading.Lock()
        self._queue_cond = threading.Condition(self._lock)
        self._queue_count = 0
        self._current_queue_bytes = 0
        self._shutdown = False
        self._close_when_drained = False
        self._pending_paths: set[str] = set()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="PresignUploader",
        )
        self._lru_uploaded: "OrderedDict[str, bool]" = OrderedDict()

        self._pid = os.getpid()
        if hasattr(os, "register_at_fork"):
            weak_reinit = weakref.WeakMethod(self._at_fork_reinit)
            os.register_at_fork(
                after_in_child=lambda: (ref := weak_reinit()) and ref()
            )

    @staticmethod
    def _parse_base_path(base_path: str) -> Tuple[str, str, str]:
        """Split ``sls://{project}/{logstore}[/{prefix}]`` into its parts.

        Objects are addressed by project/logstore/objectName because ARMS owns
        the backing bucket and decides where an object physically lands.
        """
        parsed = urlparse((base_path or "").strip())
        if parsed.scheme.lower() != "sls" or not parsed.netloc:
            raise PresignUploadConfigError(
                "Presigned multimodal storage path must use "
                "sls://project/logstore[/prefix]"
            )
        if parsed.params or parsed.query or parsed.fragment:
            raise PresignUploadConfigError(
                "Presigned multimodal storage path must not contain params, "
                "query, or fragment"
            )
        logstore, _, prefix = parsed.path.strip("/").partition("/")
        if not logstore:
            raise PresignUploadConfigError(
                "Presigned multimodal storage path is missing the logstore"
            )
        prefix = prefix.strip("/")
        if any(
            part in (".", "..")
            for part in (*logstore.split("/"), *prefix.split("/"))
            if part
        ):
            raise PresignUploadConfigError(
                "Presigned multimodal storage path must not contain dot segments"
            )
        return parsed.netloc, logstore, prefix

    @property
    def base_path(self) -> str:
        return self._base_path

    @property
    def project(self) -> str:
        return self._project

    @property
    def logstore(self) -> str:
        return self._logstore

    def _new_http_client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self._timeout,
            verify=self._ssl_verify,
            follow_redirects=False,
        )

    def _object_key_from_url(self, url: str) -> str:
        target = (url or "").strip()
        if target.startswith("sls://"):
            parsed = urlparse(target)
            logstore, _, remainder = parsed.path.lstrip("/").partition("/")
            if parsed.netloc != self._project or logstore != self._logstore:
                raise PresignUploadConfigError(
                    "Upload item does not match the configured SLS project "
                    "and logstore"
                )
            key = remainder.lstrip("/")
        else:
            key = target.lstrip("/")
            if self._prefix and not (
                key == self._prefix or key.startswith(f"{self._prefix}/")
            ):
                key = f"{self._prefix}/{key}"
        if not key:
            raise PresignUploadConfigError(
                "Upload item OSS object key is empty"
            )
        if self._prefix and not (
            key == self._prefix or key.startswith(f"{self._prefix}/")
        ):
            raise PresignUploadConfigError(
                "Upload item is outside the configured OSS prefix"
            )
        if any(part in (".", "..") for part in key.split("/")):
            raise PresignUploadConfigError("Upload item contains dot segments")
        return key

    def upload(
        self,
        item: UploadItem,
        *,
        skip_if_exists: bool = True,
    ) -> bool:
        recorder = get_multimodal_usage_recorder()
        if item.data is None and item.source_uri is None:
            recorder.record_upload_error(
                provider=_USAGE_METRICS_PROVIDER,
                reason="invalid_item",
            )
            return False

        try:
            object_key = self._object_key_from_url(item.url)
        except PresignConfigError as exc:
            _logger.warning("Invalid presigned upload item: %s", exc)
            recorder.record_upload_error(
                provider=_USAGE_METRICS_PROVIDER,
                reason="invalid_item",
            )
            return False

        content = item.data
        if isinstance(content, str):
            content = content.encode()
        full_path = f"sls://{self._project}/{self._logstore}/{object_key}"
        reserved_size = (
            len(content) if content is not None else max(0, item.expected_size)
        )
        task = _Task(
            full_path=full_path,
            object_key=object_key,
            content=content,
            source_uri=item.source_uri,
            expected_size=max(0, item.expected_size),
            reserved_size=reserved_size,
            skip_if_exists=skip_if_exists,
        )

        with self._queue_cond:
            if skip_if_exists:
                if self._uploaded_cached_locked(full_path):
                    return True
                if full_path in self._pending_paths:
                    return True
            if self._shutdown:
                recorder.record_upload_error(
                    provider=_USAGE_METRICS_PROVIDER,
                    reason="shutdown",
                )
                return False
            if self._queue_count >= self._queue_capacity:
                _logger.debug(
                    "Presigned upload queue full, dropping: %s", full_path
                )
                recorder.record_upload_error(
                    provider=_USAGE_METRICS_PROVIDER,
                    reason="queue_full",
                )
                return False
            if (
                self._max_queue_bytes > 0
                and self._current_queue_bytes + reserved_size
                > self._max_queue_bytes
            ):
                _logger.debug(
                    "Presigned upload queue bytes limit exceeded "
                    "(current=%d, incoming=%d, max=%d), dropping: %s",
                    self._current_queue_bytes,
                    reserved_size,
                    self._max_queue_bytes,
                    full_path,
                )
                recorder.record_upload_error(
                    provider=_USAGE_METRICS_PROVIDER,
                    reason="queue_bytes_limit",
                )
                return False
            if skip_if_exists:
                self._pending_paths.add(full_path)
            self._queue_count += 1
            self._current_queue_bytes += reserved_size
            try:
                self._executor.submit(self._do_upload, task)
            except RuntimeError:
                self._queue_count -= 1
                self._current_queue_bytes -= reserved_size
                if skip_if_exists:
                    self._pending_paths.discard(full_path)
                self._queue_cond.notify_all()
                recorder.record_upload_error(
                    provider=_USAGE_METRICS_PROVIDER,
                    reason="shutdown",
                )
                return False
        return True

    def _do_upload(self, task: _Task) -> None:
        recorder = get_multimodal_usage_recorder()
        succeeded = False
        try:
            if task.content is None and task.source_uri:
                task.content = self._download_content(
                    task.source_uri,
                    max_size=_MAX_DOWNLOAD_BYTES,
                )
                if task.content is None:
                    recorder.record_upload_error(
                        provider=_USAGE_METRICS_PROVIDER,
                        reason="download_failed",
                    )
                    return
                self._adjust_reserved_size(task, len(task.content))

            content = task.content
            if content is None:
                recorder.record_upload_error(
                    provider=_USAGE_METRICS_PROVIDER,
                    reason="invalid_item",
                )
                return

            reason = self._upload_with_retries(task, content)
            if reason is None:
                succeeded = True
                recorder.record_upload_success(
                    provider=_USAGE_METRICS_PROVIDER,
                    content_bytes=len(content),
                )
            else:
                recorder.record_upload_error(
                    provider=_USAGE_METRICS_PROVIDER,
                    reason=reason,
                )
        finally:
            self._release_task(task, succeeded=succeeded)

    def _upload_with_retries(
        self, task: _Task, content: bytes
    ) -> Optional[str]:
        """Upload one object; return ``None`` on success or a failure reason."""
        for attempt in range(1, self._max_upload_attempts + 1):
            last_attempt = attempt >= self._max_upload_attempts
            try:
                target = self._client.presign(task.object_key)
            except PresignAuthError as exc:
                _logger.warning(
                    "Presign request unauthorized for %s: %s",
                    task.full_path,
                    exc,
                )
                return "auth"
            except PresignRetryableError as exc:
                if last_attempt:
                    _logger.warning(
                        "Presign request failed for %s after %d attempt(s): %s",
                        task.full_path,
                        attempt,
                        exc,
                    )
                    return "network"
                self._retry_delay(attempt, task.full_path, exc)
                continue
            except PresignError as exc:
                _logger.warning(
                    "Presign request rejected for %s: %s",
                    task.full_path,
                    exc,
                )
                return "presign_error"

            try:
                self._put_presigned(target, content)
                return None
            except httpx.TransportError as exc:
                if last_attempt:
                    _logger.warning(
                        "Presigned upload network failure for %s after %d "
                        "attempt(s): %s",
                        task.full_path,
                        attempt,
                        exc,
                    )
                    return "network"
                self._retry_delay(attempt, task.full_path, exc)
            except _PresignedUploadRejected as exc:
                if exc.retryable and not last_attempt:
                    self._retry_delay(attempt, task.full_path, exc)
                    continue
                _logger.warning(
                    "Presigned upload failed for %s after %d attempt(s): %s",
                    task.full_path,
                    attempt,
                    exc,
                )
                return exc.reason
            except Exception as exc:  # pylint: disable=broad-except
                _logger.warning(
                    "Unexpected presigned upload failure for %s: %s",
                    task.full_path,
                    exc,
                )
                return "storage_error"
        return "storage_error"

    def _put_presigned(
        self,
        target: Any,
        content: bytes,
    ) -> None:
        # The ARMS presigned URL signs no request headers, and OSS folds any
        # header we add into the canonical request, so sending our own
        # Content-Type breaks the signature. Send only what ARMS asked for.
        headers: Dict[str, str] = dict(target.headers or {})

        with suppress_http_instrumentation():
            response = self._http_client.request(
                target.method or "PUT",
                target.url,
                content=content,
                headers=headers,
            )
        if response.is_success:
            return

        status = response.status_code
        if status in (401, 403):
            raise _PresignedUploadRejected(
                f"presigned upload unauthorized (status={status}): "
                f"{self._response_snippet(response)}",
                reason="auth",
                retryable=False,
            )
        if status in (408, 429) or status >= 500:
            raise _PresignedUploadRejected(
                f"presigned upload failed (status={status})",
                reason="storage_error",
                retryable=True,
            )
        raise _PresignedUploadRejected(
            f"presigned upload rejected (status={status})",
            reason="storage_error",
            retryable=False,
        )

    @staticmethod
    def _response_snippet(response: httpx.Response, limit: int = 512) -> str:
        try:
            return response.text[:limit]
        except Exception:  # pylint: disable=broad-except
            return ""

    def _retry_delay(
        self,
        attempt: int,
        full_path: str,
        error: Exception,
    ) -> None:
        maximum = self._upload_retry_delay * (2 ** (attempt - 1))
        delay = random.uniform(0.0, maximum) if maximum > 0 else 0.0
        _logger.debug(
            "Retrying presigned upload for %s after attempt %d in %.2fs: %s",
            full_path,
            attempt,
            delay,
            error,
        )
        if delay > 0:
            time.sleep(delay)

    def _adjust_reserved_size(self, task: _Task, actual_size: int) -> None:
        with self._queue_cond:
            difference = actual_size - task.reserved_size
            self._current_queue_bytes += difference
            task.reserved_size = actual_size

    def _release_task(self, task: _Task, *, succeeded: bool) -> None:
        with self._queue_cond:
            if succeeded:
                self._mark_uploaded_locked(task.full_path)
            if task.skip_if_exists:
                self._pending_paths.discard(task.full_path)
            self._queue_count -= 1
            self._current_queue_bytes -= task.reserved_size
            close_clients = self._close_when_drained and self._queue_count == 0
            self._queue_cond.notify_all()
        if close_clients:
            self._close_clients()

    def _download_content(
        self,
        uri: str,
        *,
        max_size: int,
    ) -> Optional[bytes]:
        with suppress_http_instrumentation():
            try:
                with self._http_client.stream(
                    "GET", uri, follow_redirects=False
                ) as response:
                    if 300 <= response.status_code < 400:
                        raise httpx.HTTPStatusError(
                            "Redirect not allowed",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    buffer = io.BytesIO()
                    for chunk in response.iter_bytes(chunk_size=64 * 1024):
                        if buffer.tell() + len(chunk) > max_size:
                            _logger.warning(
                                "Download exceeds multimodal limit %d, "
                                "aborting: %s",
                                max_size,
                                uri,
                            )
                            return None
                        buffer.write(chunk)
                    return buffer.getvalue()
            except Exception as exc:  # pylint: disable=broad-except
                _logger.warning(
                    "Failed to download multimodal source %s: %s", uri, exc
                )
                return None

    def _uploaded_cached(self, path: str) -> bool:
        with self._queue_cond:
            return self._uploaded_cached_locked(path)

    def _uploaded_cached_locked(self, path: str) -> bool:
        if path in self._lru_uploaded:
            self._lru_uploaded.move_to_end(path)
            return True
        return False

    def _mark_uploaded_locked(self, path: str) -> None:
        self._lru_uploaded[path] = True
        if len(self._lru_uploaded) > self._lru_capacity:
            self._lru_uploaded.popitem(last=False)

    def shutdown(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        with self._queue_cond:
            if self._shutdown:
                return
            self._shutdown = True
            while self._queue_count > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _logger.warning(
                        "Presigned uploader shutdown timed out with %d task(s)",
                        self._queue_count,
                    )
                    break
                self._queue_cond.wait(timeout=remaining)
            drained = self._queue_count == 0
            # Active requests keep their clients until the final task finishes.
            self._close_when_drained = not drained
        self._executor.shutdown(wait=False)
        if drained:
            self._close_clients()

    def _close_clients(self) -> None:
        # Called once: by shutdown when drained, otherwise by the last task.
        try:
            if self._owns_http_client:
                self._http_client.close()
        finally:
            self._client.close()

    def _at_fork_reinit(self) -> None:
        self._lock = threading.Lock()
        self._queue_cond = threading.Condition(self._lock)
        self._queue_count = 0
        self._current_queue_bytes = 0
        self._shutdown = False
        self._close_when_drained = False
        self._pending_paths = set()
        self._lru_uploaded = OrderedDict()
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="PresignUploader",
        )
        if self._owns_http_client:
            self._http_client = self._new_http_client()
        self._pid = os.getpid()


def _resolve_snapshot(snapshot: Optional[Any]) -> Any:
    if snapshot is not None:
        return snapshot
    from opentelemetry.util.genai._multimodal_upload.config import (  # pylint: disable=import-outside-toplevel,no-name-in-module  # noqa: PLC0415
        get_multimodal_config_snapshot,
    )

    return get_multimodal_config_snapshot()


def _resolve_sls_base_path(snapshot: Any) -> str:
    """Return the ``sls://{project}/{logstore}[/{prefix}]`` base path.

    The span records this logical address before the upload starts, so it must
    be derivable without calling ARMS: the server decides which bucket backs
    the object, but always stores it under ``{project}/{logstore}/{name}``.
    """
    project, logstore = _resolve_sls_target(snapshot)
    base_path = format_sls_base_path(
        project, logstore, getattr(snapshot, "oss_path_prefix", None)
    )
    if not base_path:
        raise PresignUploadConfigError(
            "Pre-authorized OSS mode requires an SLS project, set "
            "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_PROJECT or run with an ARMS "
            "agent that provides one"
        )
    return base_path


def _resolve_sls_target(snapshot: Any) -> Tuple[str, str]:
    """Resolve the project/logstore pair that addresses uploaded objects.

    The logstore always falls back to the shared default so the recorded URI
    and the presign request agree on where the object lands.
    """
    project = (getattr(snapshot, "sls_project", None) or "").strip()
    logstore = (
        getattr(snapshot, "sls_logstore", None) or ""
    ).strip() or DEFAULT_SLS_LOGSTORE
    if project:
        return project, logstore
    try:
        from aliyun.sdk.extension.arms.exporters.arms_endpoints_state import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
            global_arms_endpoints_state,
        )

        project = (global_arms_endpoints_state.sls_project or "").strip()
    except Exception:  # pylint: disable=broad-except
        _logger.debug(
            "ARMS endpoint state unavailable for multimodal presign project",
            exc_info=True,
        )
    return project, logstore


def build_presign_client(snapshot: Any) -> MultimodalPresignClient:
    """Create a presign client for the configured project and logstore."""
    license_key = (
        getattr(snapshot, "presign_license_key", None) or ""
    ).strip()
    workspace = (getattr(snapshot, "presign_workspace", None) or "").strip()
    project, logstore = _resolve_sls_target(snapshot)
    timeout = _presign_timeout()
    ssl_verify = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY,
        "true",
    ).lower() not in ("false", "0", "no")
    endpoint = resolve_presign_endpoint(snapshot)
    if not endpoint:
        raise PresignConfigError(
            "Pre-authorized OSS mode could not resolve the ARMS presign endpoint"
        )
    return MultimodalPresignClient(
        license_key=license_key,
        workspace=workspace,
        project=project,
        logstore=logstore,
        endpoint_provider=lambda: resolve_presign_endpoint(snapshot),
        timeout=timeout,
        verify=ssl_verify,
    )


def presign_uploader_hook(
    snapshot: Optional[Any] = None,
) -> Optional[Uploader]:
    """Build the pre-authorized OSS uploader from the runtime snapshot."""
    cfg = _resolve_snapshot(snapshot)
    try:
        base_path = _resolve_sls_base_path(cfg)
        client = build_presign_client(cfg)
        return PresignUploader(
            base_path,
            client=client,
            timeout=_presign_timeout(),
        )
    except (PresignConfigError, ValueError) as exc:
        _logger.warning(
            "Pre-authorized OSS multimodal uploader disabled: %s", exc
        )
        return None


def presign_pre_uploader_hook(
    snapshot: Optional[Any] = None,
) -> Optional[PreUploader]:
    """Build the generic pre-uploader only when presigned upload can start."""
    cfg = _resolve_snapshot(snapshot)
    try:
        base_path = _resolve_sls_base_path(cfg)
        # Keep the uploader pair atomic: never rewrite inline media into
        # sls:// URIs when the matching uploader cannot be built.
        if not (getattr(cfg, "presign_license_key", None) or "").strip():
            raise PresignConfigError(
                "Pre-authorized OSS mode requires an ARMS license key"
            )
        if not resolve_presign_endpoint(cfg):
            raise PresignConfigError(
                "Pre-authorized OSS mode could not resolve the ARMS presign "
                "endpoint"
            )
        from opentelemetry.util.genai._multimodal_upload.pre_uploader import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
            MultimodalPreUploader,
        )

        return MultimodalPreUploader(base_path=base_path)
    except (ImportError, PresignConfigError, ValueError) as exc:
        _logger.warning(
            "Pre-authorized OSS multimodal pre-uploader disabled: %s", exc
        )
        return None
