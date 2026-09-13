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

import json
import re
from pathlib import Path

import pytest

_CASSETTES = Path(__file__).parent / "cassettes"
_THOUGHT_SIGNATURE = re.compile(
    rb'("(?:thoughtSignature|thought_signature|signature)"\s*:\s*")[^"]+(")'
)
_GENERATED_RESULT_SCHEMA_KEYS = {"response_json_schema", "responseJsonSchema"}
_GENERATED_PARAMETER_SCHEMA_KEYS = {
    "parameters_json_schema",
    "parametersJsonSchema",
}


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


def _normalize_generated_schema(value):
    """Normalize SDK-generated function schemas across google-genai 2.x."""
    if isinstance(value, dict):
        normalized = {}
        for key, nested in value.items():
            if key in _GENERATED_RESULT_SCHEMA_KEYS:
                continue
            normalized_key = (
                "parameters"
                if key in _GENERATED_PARAMETER_SCHEMA_KEYS
                else key
            )
            normalized_value = _normalize_generated_schema(nested)
            # google-genai 2.23 switched generated JSON Schema type values
            # from the API's uppercase spelling to standard lowercase.
            if (
                normalized_key == "type"
                and isinstance(nested, str)
                and nested.upper()
                in {
                    "ARRAY",
                    "BOOLEAN",
                    "INTEGER",
                    "NUMBER",
                    "OBJECT",
                    "STRING",
                }
            ):
                normalized_value = nested.lower()
            normalized[normalized_key] = normalized_value
        return normalized
    if isinstance(value, list):
        return [_normalize_generated_schema(item) for item in value]
    return value


def _normalize_request_body(value):
    """Canonicalize JSON requests while retaining exact semantic matching."""
    scrubbed = _scrub_body(value)
    is_bytes = isinstance(scrubbed, bytes)
    try:
        payload = json.loads(scrubbed)
    except (TypeError, ValueError):
        return scrubbed
    canonical = json.dumps(
        _normalize_generated_schema(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    return canonical.encode() if is_bytes else canonical


def _scrub_request(request):
    request.body = _normalize_request_body(request.body)
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
