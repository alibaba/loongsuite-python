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

"""Multimodal Upload Base Interfaces

This module defines abstract interfaces and data types for uploaders, for implementation classes to inherit from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from opentelemetry.trace import SpanContext


@dataclass
class UploadItem:
    """Upload task item

    Distinguish task types via data and source_uri:
    - data has value: direct upload (BlobPart)
    - source_uri has value: download then upload (UriPart)
    """

    url: str  # Target upload path
    content_type: str
    meta: Dict[str, str]
    data: Optional[bytes] = None  # Data for direct upload
    source_uri: Optional[str] = None  # Source URI, download then upload
    expected_size: int = 0  # Estimated size, for queue management


# Backward compatible alias
PreUploadItem = UploadItem


class Uploader(ABC):
    """File uploader abstract base class

    Implementer notes:
    - upload method should be non-blocking (async enqueue)
    - Implementation should be idempotent: duplicate uploads to the same path should be skipped
    - Implementation should handle internal exceptions, not throw to caller
    """

    @abstractmethod
    def upload(
        self,
        item: UploadItem,
        *,
        skip_if_exists: bool = True,
    ) -> bool:
        """Enqueue a file upload.

        Args:
            item: Upload task item, including url, data/source_uri, content_type, meta, etc.
            skip_if_exists: Skip if file already exists

        Returns:
            True: Task successfully enqueued or already exists (skipped)
            False: Queue is full or uploader is shutting down

        Note:
            - This method does not throw exceptions, returns False on failure
            - Actual upload executes asynchronously in background threads
            - Upload failures are logged internally, do not affect caller
        """

    @abstractmethod
    def shutdown(self, timeout: float = 10.0) -> None:
        """Gracefully shutdown the uploader.

        Args:
            timeout: Maximum wait time (seconds)
        """


class PreUploader(ABC):
    """Multimodal data pre-upload processor abstract base class

    Implementer notes:
    - pre_upload is a synchronous method, but may contain async operations internally
    - Implementation should handle internal exceptions, return empty list rather than throw
    - Implementation should modify the passed messages object (in-place replacement)
    """

    @abstractmethod
    def pre_upload(
        self,
        span_context: Optional[SpanContext],
        start_time_utc_nano: int,
        input_messages: Optional[List[Any]],
        output_messages: Optional[List[Any]],
        config_snapshot: Optional[Any] = None,
    ) -> List[PreUploadItem]:
        """
        Preprocess multimodal data in messages:
        - Process BlobPart/Blob (base64 inline data) and UriPart/Uri (external references)
        - Generate complete URL: {base_path}/{date}/{md5}.{ext}
        - Replace the original part with UriPart pointing to uploaded URL
        - Return the list of data to be uploaded

        Args:
            span_context: Span context for trace/span IDs
            start_time_utc_nano: Start time in nanoseconds
            input_messages: List of input messages (with .parts attribute), will be modified in-place
            output_messages: List of output messages (with .parts attribute), will be modified in-place
            config_snapshot: Optional runtime config snapshot; when omitted, implementation
                may read process-wide multimodal config itself

        Returns:
            List of PreUploadItem to be uploaded, returns empty list on failure

        Note:
            - This method does not throw exceptions, returns empty list on failure and logs
            - Passed messages will be modified in-place, replacing BlobPart with UriPart
            - Returned PreUploadItem needs to be uploaded via Uploader.upload()
        """

    def shutdown(  # pylint: disable=no-self-use,unused-argument,useless-return
        self, timeout: float = 5.0
    ) -> None:
        """Gracefully shutdown the pre-uploader.

        Default implementation is no-op for lightweight/stateless implementations.

        Args:
            timeout: Maximum wait time (seconds)
        """
        # Default no-op for stateless implementations.
        return None
