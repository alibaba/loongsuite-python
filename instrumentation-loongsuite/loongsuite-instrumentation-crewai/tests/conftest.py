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

import os

import pytest

# litellm fetches its model cost map from raw.githubusercontent.com the first time
# it is used. Under ``record_mode=none`` that request has no recorded counterpart,
# so VCR reports "Can't overwrite existing cassette" and its method/host/path
# matchers fail against the recorded api.openai.com interactions. Loading the
# bundled copy keeps the suite offline; litellm documents the switch in
# ``litellm.litellm_core_utils.get_model_cost_map``.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def scrub_response_headers(response):
    """Clean up common sensitive response headers before recording."""
    headers = response.get("headers", {})
    for secret_header in [
        "Set-Cookie",
        "x-request-id",
        "x-amz-request-id",
        "x-goog-hash",
    ]:
        if secret_header in headers:
            headers[secret_header] = ["<masked>"]
    return response


@pytest.fixture(scope="module")
def vcr_config():
    """VCR configuration to mask API keys and sensitive information."""
    return {
        "filter_headers": [
            ("authorization", "Bearer <masked>"),
            ("api-key", "<masked>"),
            ("api_key", "<masked>"),
            ("x-api-key", "<masked>"),
            ("cookie", "<masked>"),
        ],
        "filter_query_parameters": [
            ("api_key", "<masked>"),
        ],
        "filter_post_data_parameters": [
            ("api_key", "<masked>"),
        ],
        "before_record_response": scrub_response_headers,
        "decode_compressed_response": True,
        # Requests incidental to the code under test must not reach the
        # cassettes, or they fail playback under ``record_mode=none``. The
        # litellm package's own conftest ignores its telemetry hosts for the
        # same reason.
        "ignore_hosts": [
            "raw.githubusercontent.com",
            "us.i.posthog.com",
            "app.posthog.com",
            "api.litellm.ai",
            "telemetry.crewai.com",
        ],
    }
