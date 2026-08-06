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

from opentelemetry.instrumentation.google_genai.client_info import (
    get_client_info,
)


def _client(server_url: str) -> SimpleNamespace:
    return SimpleNamespace(
        sdk_configuration=SimpleNamespace(server_url=server_url)
    )


@pytest.mark.parametrize(
    "server_url, expected_address",
    [
        (
            "https://aiplatform.googleapis.com/v1",
            "aiplatform.googleapis.com",
        ),
        (
            "https://us-central1-aiplatform.googleapis.com/v1",
            "us-central1-aiplatform.googleapis.com",
        ),
        (
            "HTTPS://AIPLATFORM.GOOGLEAPIS.COM/v1",
            "aiplatform.googleapis.com",
        ),
        (
            "us-central1-aiplatform.googleapis.com",
            "us-central1-aiplatform.googleapis.com",
        ),
    ],
)
def test_vertex_hostname_is_recognized(
    server_url: str,
    expected_address: str,
) -> None:
    is_vertex, server_address = get_client_info(_client(server_url))

    assert is_vertex
    assert server_address == expected_address


@pytest.mark.parametrize(
    "server_url, expected_address",
    [
        (
            "https://aiplatform.googleapis.com.evil.example/v1",
            "aiplatform.googleapis.com.evil.example",
        ),
        (
            "https://evil.example/v1?next=aiplatform.googleapis.com",
            "evil.example",
        ),
        (
            "https://aiplatform.googleapis.com@evil.example/v1",
            "evil.example",
        ),
        (
            "aiplatform.googleapis.com.evil.example",
            "aiplatform.googleapis.com.evil.example",
        ),
    ],
)
def test_vertex_hostname_rejects_substring_confusion(
    server_url: str,
    expected_address: str,
) -> None:
    is_vertex, server_address = get_client_info(_client(server_url))

    assert not is_vertex
    assert server_address == expected_address
