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
_recorder: Optional["MultimodalUsageRecorder"] = None


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
    def record_upload_success(
        self, *, provider: str, content_bytes: int
    ) -> None:
        return

    def record_upload_error(self, *, provider: str, reason: str) -> None:
        return


_DEFAULT_RECORDER = _NoOpMultimodalUsageRecorder()


def get_multimodal_usage_recorder() -> MultimodalUsageRecorder:
    """Return the current recorder, or the default no-op implementation."""
    with _lock:
        if _recorder is None:
            return _DEFAULT_RECORDER
        return _recorder


def set_multimodal_usage_recorder(
    recorder: Optional[MultimodalUsageRecorder],
) -> None:
    """Replace the global recorder. ``None`` resets to the default no-op."""
    global _recorder
    with _lock:
        _recorder = recorder


def provider_label_from_protocol(protocol: str) -> str:
    """Map fsspec protocol to a low-cardinality provider label."""
    normalized = (protocol or "").lower()
    if normalized == "oss":
        return "oss"
    if normalized == "sls":
        return "sls"
    return "other"
