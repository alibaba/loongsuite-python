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

"""Multimodal upload usage recorder protocol and default no-op implementation."""

from __future__ import annotations

import logging
import threading
from typing import Optional, Protocol, runtime_checkable

_logger = logging.getLogger(__name__)

_lock = threading.Lock()
_recorder: Optional["_SafeMultimodalUsageRecorder"] = None


@runtime_checkable
class MultimodalUsageRecorder(Protocol):
    """Records multimodal upload success and terminal failure metrics."""

    def record_upload_success(
        self, *, provider: str, content_bytes: int
    ) -> None:
        """Record one successful object upload."""

    def record_upload_error(self, *, provider: str, reason: str) -> None:
        """Record one terminal upload failure."""


class _NoOpMultimodalUsageRecorder:
    def record_upload_success(  # pylint: disable=no-self-use
        self, *, provider: str, content_bytes: int
    ) -> None:
        return

    def record_upload_error(  # pylint: disable=no-self-use
        self, *, provider: str, reason: str
    ) -> None:
        return


class _SafeMultimodalUsageRecorder:
    """Proxy that swallows recorder exceptions so callers are never disrupted."""

    def __init__(self, inner: MultimodalUsageRecorder) -> None:
        self._inner = inner

    def record_upload_success(
        self, *, provider: str, content_bytes: int
    ) -> None:
        try:
            self._inner.record_upload_success(
                provider=provider, content_bytes=content_bytes
            )
        except Exception:  # pylint: disable=broad-except
            _logger.debug(
                "Multimodal usage recorder record_upload_success failed",
                exc_info=True,
            )

    def record_upload_error(self, *, provider: str, reason: str) -> None:
        try:
            self._inner.record_upload_error(provider=provider, reason=reason)
        except Exception:  # pylint: disable=broad-except
            _logger.debug(
                "Multimodal usage recorder record_upload_error failed",
                exc_info=True,
            )


_DEFAULT_RECORDER = _SafeMultimodalUsageRecorder(
    _NoOpMultimodalUsageRecorder()
)


def get_multimodal_usage_recorder() -> MultimodalUsageRecorder:
    """Return the current recorder, or the default no-op implementation.

    The returned recorder never propagates exceptions from the underlying
    implementation.
    """
    with _lock:
        if _recorder is None:
            return _DEFAULT_RECORDER
        return _recorder


def set_multimodal_usage_recorder(
    recorder: Optional[MultimodalUsageRecorder],
) -> None:
    """Replace the global recorder. ``None`` resets to the default no-op."""
    global _recorder  # pylint: disable=global-statement
    with _lock:
        if recorder is None:
            _recorder = None
        else:
            _recorder = _SafeMultimodalUsageRecorder(recorder)


def provider_label_from_protocol(protocol: str) -> str:
    """Map fsspec protocol to a low-cardinality provider label."""
    normalized = (protocol or "").lower()
    if normalized == "oss":
        return "oss"
    if normalized == "sls":
        return "sls"
    return "other"
