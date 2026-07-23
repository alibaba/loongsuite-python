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

"""VCR configuration for Google GenAI SDK integration tests."""

import re
from pathlib import Path

import pytest

_CASSETTES = Path(__file__).parent / "cassettes"
_THOUGHT_SIGNATURE = re.compile(
    rb'("(?:thoughtSignature|thought_signature)"\s*:\s*")[^"]+(")'
)


def _scrub_body(value):
    if isinstance(value, str):
        return _THOUGHT_SIGNATURE.sub(
            rb"\1dGVzdC10aG91Z2h0LXNpZ25hdHVyZQ==\2", value.encode()
        ).decode()
    if isinstance(value, bytes):
        return _THOUGHT_SIGNATURE.sub(
            rb"\1dGVzdC10aG91Z2h0LXNpZ25hdHVyZQ==\2", value
        )
    return value


def _scrub_request(request):
    request.body = _scrub_body(request.body)
    return request


def _scrub_response(response):
    body = response.get("body", {})
    if "string" in body:
        body["string"] = _scrub_body(body["string"])
    headers = response.get("headers", {})
    for name in list(headers):
        if name.lower() in {
            "date",
            "server",
            "server-timing",
            "set-cookie",
            "x-request-id",
        }:
            del headers[name]
    return response


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(_CASSETTES)


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            ("authorization", "Bearer test_google_key"),
            ("x-goog-api-key", "test-google-key"),
        ],
        "filter_query_parameters": [
            "key",
            "apiKey",
            "quotaUser",
            "userProject",
            "access_token",
        ],
        "filter_post_data_parameters": ["key", "api_key", "apiKey"],
        "before_record_request": _scrub_request,
        "before_record_response": _scrub_response,
        "decode_compressed_response": True,
        "match_on": [
            "method",
            "scheme",
            "host",
            "port",
            "path",
            "query",
            "body",
        ],
    }
