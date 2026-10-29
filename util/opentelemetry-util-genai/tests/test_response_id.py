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

from types import SimpleNamespace

import pytest

from opentelemetry.util.genai.response_id import (
    extract_response_id,
    resolve_response_id,
)


@pytest.mark.parametrize(
    ("response", "fields", "expected"),
    [
        ("  chatcmpl-123  ", ("id",), "chatcmpl-123"),
        (123, ("id",), "123"),
        ({"request_id": "dashscope-123"}, ("request_id",), "dashscope-123"),
        (SimpleNamespace(id="msg-123"), ("id",), "msg-123"),
        (
            {"id": "", "response_id": "resp-123"},
            ("id", "response_id"),
            "resp-123",
        ),
        (SimpleNamespace(id=None), ("id",), None),
        (True, ("id",), None),
    ],
)
def test_extract_response_id(response, fields, expected):
    assert extract_response_id(response, fields=fields) == expected


def test_extract_response_id_uses_caller_field_order():
    response = {"id": "completion-123", "request_id": "request-123"}

    assert (
        extract_response_id(response, fields=("request_id", "id"))
        == "request-123"
    )
    assert (
        extract_response_id(response, fields=("id", "request_id"))
        == "completion-123"
    )


def test_extract_response_id_does_not_use_transport_id_implicitly():
    response = SimpleNamespace(_request_id="transport-request-123")

    assert extract_response_id(response) is None


def test_resolve_response_id_prefers_provider_and_falls_back_to_framework():
    framework_response = SimpleNamespace(id="stream-framework-123")

    assert (
        resolve_response_id("chatcmpl-provider-123", framework_response)
        == "chatcmpl-provider-123"
    )
    assert (
        resolve_response_id(None, framework_response) == "stream-framework-123"
    )


def test_extract_response_id_ignores_raising_sdk_property():
    class RaisingResponse:
        @property
        def request_id(self):
            raise RuntimeError("not loaded")

        id = "chatcmpl-after-error"

    assert (
        extract_response_id(
            RaisingResponse(),
            fields=("request_id", "id"),
        )
        == "chatcmpl-after-error"
    )
