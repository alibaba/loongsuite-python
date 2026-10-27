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
from pathlib import Path

import pytest

# litellm fetches its model cost map from raw.githubusercontent.com the first time
# it is used. Under ``record_mode=none`` that request has no recorded counterpart,
# so VCR reports "Can't overwrite existing cassette" and its method/host/path
# matchers fail against the recorded api.openai.com interactions. Loading the
# bundled copy keeps the suite offline; litellm documents the switch in
# ``litellm.litellm_core_utils.get_model_cost_map``.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# crewAI exports anonymous telemetry to telemetry.crewai.com from a background
# thread, and checks PyPI for a newer release when a verbose crew starts.
# Neither belongs in these offline tests. Passing the telemetry requests
# through VCR (``ignore_hosts``) made them flaky: while vcrpy sends such a
# request to the real host, ``vcr.patch.force_reset`` undoes its patches for
# every HTTP client in the process, so an LLM call crewAI made meanwhile on its
# worker thread skipped the cassette and reached the real provider. When that
# window spanned the end of a test, the patches went back out of order and the
# next test was replayed from the previous test's cassette.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_VERSION_CHECK"] = "true"

_TESTS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items):
    """Block real network access in these tests.

    Every response they need is recorded in the cassettes, so a connection
    attempt means a request got past VCR: pytest-recording's
    ``block_network`` mark makes it fail with "Network is disabled" instead
    of reaching a real endpoint. As with ``--block-network``, recording new
    cassettes (``--record-mode`` other than ``none``) still has network
    access.
    """
    for item in items:
        if item.path.is_relative_to(_TESTS_DIR):
            item.add_marker(pytest.mark.block_network)


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
        # No ``ignore_hosts``: vcrpy sends requests for ignored hosts to the
        # real network, and while it does, concurrent LLM calls skip the
        # cassette (see the telemetry switches above).
    }
