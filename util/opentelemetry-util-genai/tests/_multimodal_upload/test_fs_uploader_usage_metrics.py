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

"""FsUploader multimodal usage metrics unit tests."""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

from opentelemetry.util.genai._multimodal_upload import FsUploader, UploadItem
from opentelemetry.util.genai._multimodal_upload.fs_uploader import _Task
from opentelemetry.util.genai._multimodal_upload.usage_recorder import (
    set_multimodal_usage_recorder,
)


@dataclass
class RecordingUsageRecorder:
    successes: List[tuple[str, int]] = field(default_factory=list)
    errors: List[tuple[str, str]] = field(default_factory=list)
    success_event: threading.Event = field(default_factory=threading.Event)
    raise_on_success: bool = False
    raise_on_error: bool = False
    success_calls: int = 0

    def record_upload_success(
        self, *, provider: str, content_bytes: int
    ) -> None:
        self.success_calls += 1
        if self.raise_on_success:
            raise RuntimeError("boom")
        self.successes.append((provider, content_bytes))
        self.success_event.set()

    def record_upload_error(self, *, provider: str, reason: str) -> None:
        if self.raise_on_error:
            raise RuntimeError("boom")
        self.errors.append((provider, reason))


@pytest.fixture(autouse=True)
def reset_recorder():
    set_multimodal_usage_recorder(None)
    yield
    set_multimodal_usage_recorder(None)


def test_success_records_count_and_bytes_once():
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "upload_bytes_test"))
    os.makedirs(base_dir, exist_ok=True)
    recorder = RecordingUsageRecorder()
    set_multimodal_usage_recorder(recorder)

    content = b"hello-multimodal"
    uploader = FsUploader(base_path=base_dir, max_workers=1)
    try:
        dst = os.path.join(base_dir, "usage-success.bin")
        if os.path.exists(dst):
            os.remove(dst)

        assert uploader.upload(
            UploadItem(
                url="usage-success.bin",
                data=content,
                content_type="application/octet-stream",
                meta={},
            )
        )
        uploader.shutdown(timeout=5.0)
    finally:
        uploader.shutdown(timeout=1.0)

    assert recorder.successes == [("other", len(content))]
    assert recorder.errors == []


@patch(
    "opentelemetry.util.genai._multimodal_upload.fs_uploader.fsspec.url_to_fs"
)
def test_oss_success_records_oss_provider(
    mock_url_to_fs: MagicMock,
) -> None:
    mock_fs = MagicMock()
    mock_fs.unstrip_protocol.return_value = "test-bucket"
    mock_fs.exists.return_value = False
    mock_url_to_fs.return_value = (mock_fs, "test-bucket")

    recorder = RecordingUsageRecorder()
    set_multimodal_usage_recorder(recorder)

    content = b"oss-object-bytes"
    uploader = FsUploader(base_path="oss://test-bucket", max_workers=1)
    try:
        assert uploader.upload(
            UploadItem(
                url="usage-oss.bin",
                data=content,
                content_type="application/octet-stream",
                meta={"from": "test"},
            )
        )
        assert recorder.success_event.wait(timeout=5.0)
        uploader.shutdown(timeout=5.0)
    finally:
        uploader.shutdown(timeout=1.0)

    assert recorder.successes == [("oss", len(content))]
    assert recorder.errors == []
    mock_fs.pipe_file.assert_called_once()


def test_lru_cache_hit_does_not_record_success():
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "upload_bytes_test"))
    os.makedirs(base_dir, exist_ok=True)
    recorder = RecordingUsageRecorder()
    set_multimodal_usage_recorder(recorder)

    content = b"cached-content"
    uploader = FsUploader(base_path=base_dir, max_workers=1)
    try:
        dst = os.path.join(base_dir, "cached.bin")
        if os.path.exists(dst):
            os.remove(dst)
        item = UploadItem(
            url="cached.bin",
            data=content,
            content_type="application/octet-stream",
            meta={},
        )
        assert uploader.upload(item)
        assert recorder.success_event.wait(timeout=5.0)
        assert recorder.successes == [("other", len(content))]

        recorder.successes.clear()
        recorder.success_event.clear()
        assert uploader.upload(item)
        time.sleep(0.05)
    finally:
        uploader.shutdown(timeout=1.0)

    assert recorder.successes == []
    assert recorder.errors == []


def test_queue_full_records_error():
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "upload_queue_test"))
    os.makedirs(base_dir, exist_ok=True)
    recorder = RecordingUsageRecorder()
    set_multimodal_usage_recorder(recorder)

    uploader = FsUploader(base_path=base_dir, max_workers=1, max_queue_size=1)
    block_event = threading.Event()
    original_do_upload = uploader._do_upload

    def slow_do_upload(task: _Task) -> None:
        block_event.wait(timeout=5.0)
        original_do_upload(task)

    uploader._do_upload = slow_do_upload

    try:
        assert uploader.upload(
            UploadItem(
                url="blocked.bin",
                data=b"1",
                content_type="application/octet-stream",
                meta={},
            )
        )
        assert (
            uploader.upload(
                UploadItem(
                    url="dropped.bin",
                    data=b"2",
                    content_type="application/octet-stream",
                    meta={},
                )
            )
            is False
        )
    finally:
        block_event.set()
        uploader.shutdown(timeout=5.0)

    assert ("other", "queue_full") in recorder.errors


def test_invalid_item_records_error():
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "upload_queue_test"))
    os.makedirs(base_dir, exist_ok=True)
    recorder = RecordingUsageRecorder()
    set_multimodal_usage_recorder(recorder)

    uploader = FsUploader(base_path=base_dir, max_workers=1)
    try:
        assert (
            uploader.upload(
                UploadItem(
                    url="invalid.bin",
                    content_type="application/octet-stream",
                    meta={},
                )
            )
            is False
        )
    finally:
        uploader.shutdown(timeout=1.0)

    assert recorder.errors == [("other", "invalid_item")]


def test_download_failed_records_error():
    base_dir = os.path.abspath(
        os.path.join(os.getcwd(), "upload_download_exc_test")
    )
    os.makedirs(base_dir, exist_ok=True)
    recorder = RecordingUsageRecorder()
    set_multimodal_usage_recorder(recorder)

    uploader = FsUploader(base_path=base_dir, max_workers=1)
    try:
        with patch(
            "httpx.Client", side_effect=httpx.ConnectError("Network error")
        ):
            assert uploader.upload(
                UploadItem(
                    url="remote.bin",
                    source_uri="https://example.com/remote.bin",
                    expected_size=10,
                    content_type="application/octet-stream",
                    meta={},
                )
            )
            uploader.shutdown(timeout=5.0)
    finally:
        uploader.shutdown(timeout=1.0)

    assert recorder.errors == [("other", "download_failed")]


def test_storage_error_after_retry_exhausted():
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "upload_queue_test"))
    os.makedirs(base_dir, exist_ok=True)
    recorder = RecordingUsageRecorder()
    set_multimodal_usage_recorder(recorder)

    uploader = FsUploader(
        base_path=base_dir,
        max_workers=1,
        max_upload_retries=1,
        upload_retry_delay=0.01,
    )
    try:
        with patch.object(
            uploader,
            "_write_file_with_optional_headers",
            side_effect=OSError("disk full"),
        ):
            assert uploader.upload(
                UploadItem(
                    url="fail.bin",
                    data=b"x",
                    content_type="application/octet-stream",
                    meta={},
                )
            )
            uploader.shutdown(timeout=5.0)
    finally:
        uploader.shutdown(timeout=1.0)

    assert recorder.errors == [("other", "storage_error")]


def test_remote_uri_success_uses_downloaded_bytes():
    base_dir = os.path.abspath(
        os.path.join(os.getcwd(), "upload_source_uri_test")
    )
    os.makedirs(base_dir, exist_ok=True)
    recorder = RecordingUsageRecorder()
    set_multimodal_usage_recorder(recorder)

    downloaded = b"downloaded-bytes"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes.return_value = [downloaded]
    mock_response.raise_for_status = MagicMock()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    uploader = FsUploader(base_path=base_dir, max_workers=1)
    try:
        dst = os.path.join(base_dir, "remote-success.bin")
        if os.path.exists(dst):
            os.remove(dst)
        with patch("httpx.Client", return_value=mock_client):
            assert uploader.upload(
                UploadItem(
                    url="remote-success.bin",
                    source_uri="https://example.com/remote-success.bin",
                    expected_size=999,
                    content_type="application/octet-stream",
                    meta={},
                )
            )
            assert recorder.success_event.wait(timeout=5.0)
            uploader.shutdown(timeout=5.0)
    finally:
        uploader.shutdown(timeout=1.0)

    assert recorder.successes == [("other", len(downloaded))]


def test_recorder_success_exception_does_not_retry_upload():
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "upload_bytes_test"))
    os.makedirs(base_dir, exist_ok=True)
    recorder = RecordingUsageRecorder(raise_on_success=True)
    set_multimodal_usage_recorder(recorder)

    content = b"success-then-recorder-boom"
    write_calls = {"count": 0}
    uploader = FsUploader(
        base_path=base_dir,
        max_workers=1,
        max_upload_retries=3,
        upload_retry_delay=0.01,
    )
    original_write = uploader._write_file_with_optional_headers

    def counting_write(
        path: str,
        content: bytes,
        content_type: Optional[str],
        meta: Optional[dict[str, str]],
    ) -> bool:
        write_calls["count"] += 1
        return original_write(path, content, content_type, meta)

    uploader._write_file_with_optional_headers = counting_write
    dst = os.path.join(base_dir, "recorder-raise-success.bin")
    if os.path.exists(dst):
        os.remove(dst)
    try:
        assert uploader.upload(
            UploadItem(
                url="recorder-raise-success.bin",
                data=content,
                content_type="application/octet-stream",
                meta={},
            )
        )
        uploader.shutdown(timeout=5.0)
    finally:
        uploader.shutdown(timeout=1.0)

    assert write_calls["count"] == 1
    assert recorder.success_calls == 1
    assert recorder.successes == []
    assert recorder.errors == []
    assert os.path.exists(dst)


def test_recorder_error_exception_does_not_break_upload_api():
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "upload_queue_test"))
    os.makedirs(base_dir, exist_ok=True)
    recorder = RecordingUsageRecorder(raise_on_error=True)
    set_multimodal_usage_recorder(recorder)

    uploader = FsUploader(base_path=base_dir, max_workers=1)
    try:
        assert (
            uploader.upload(
                UploadItem(
                    url="invalid.bin",
                    content_type="application/octet-stream",
                    meta={},
                )
            )
            is False
        )
    finally:
        uploader.shutdown(timeout=1.0)

    assert recorder.errors == []
