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

"""Multimodal usage recorder unit tests."""

import threading
from dataclasses import dataclass, field
from typing import Iterator, List

import pytest

from opentelemetry.util.genai._multimodal_upload.usage_recorder import (
    get_multimodal_usage_recorder,
    provider_label_from_protocol,
    set_multimodal_usage_recorder,
)


@dataclass
class RecordingUsageRecorder:
    successes: List[tuple[str, int]] = field(default_factory=list)
    errors: List[tuple[str, str]] = field(default_factory=list)

    def record_upload_success(
        self, *, provider: str, content_bytes: int
    ) -> None:
        self.successes.append((provider, content_bytes))

    def record_upload_error(self, *, provider: str, reason: str) -> None:
        self.errors.append((provider, reason))


@pytest.fixture(autouse=True)
def reset_recorder() -> Iterator[None]:
    set_multimodal_usage_recorder(None)
    yield
    set_multimodal_usage_recorder(None)


def test_default_recorder_is_no_op() -> None:
    recorder = get_multimodal_usage_recorder()
    recorder.record_upload_success(provider="other", content_bytes=10)
    recorder.record_upload_error(provider="other", reason="shutdown")


def test_set_and_get_recorder() -> None:
    custom = RecordingUsageRecorder()
    set_multimodal_usage_recorder(custom)
    get_multimodal_usage_recorder().record_upload_success(
        provider="oss", content_bytes=42
    )
    assert custom.successes == [("oss", 42)]


def test_get_recorder_swallows_inner_exceptions() -> None:
    class RaisingUsageRecorder:
        def record_upload_success(
            self, *, provider: str, content_bytes: int
        ) -> None:
            raise RuntimeError("boom-success")

        def record_upload_error(self, *, provider: str, reason: str) -> None:
            raise RuntimeError("boom-error")

    set_multimodal_usage_recorder(RaisingUsageRecorder())
    recorder = get_multimodal_usage_recorder()
    recorder.record_upload_success(provider="oss", content_bytes=1)
    recorder.record_upload_error(provider="oss", reason="queue_full")


def test_set_recorder_thread_safe() -> None:
    custom = RecordingUsageRecorder()
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        set_multimodal_usage_recorder(custom)
        get_multimodal_usage_recorder().record_upload_error(
            provider="sls", reason="queue_full"
        )

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(custom.errors) == 2


@pytest.mark.parametrize(
    ("protocol", "expected"),
    [
        ("oss", "oss"),
        ("OSS", "oss"),
        ("sls", "sls"),
        ("file", "other"),
        ("", "other"),
    ],
)
def test_provider_label_from_protocol(protocol: str, expected: str) -> None:
    assert provider_label_from_protocol(protocol) == expected
