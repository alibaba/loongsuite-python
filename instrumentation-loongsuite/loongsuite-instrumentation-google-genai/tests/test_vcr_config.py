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

"""Regression tests for Google GenAI VCR request normalization."""

import json


def test_normalizer_only_rewrites_generated_function_schema(
    vcr_request_body_normalizer,
):
    generation_config = {
        "response_json_schema": {"type": "STRING"},
        "responseMimeType": "application/json",
    }
    request = {
        "contents": [{"parts": [{"text": "STRING"}], "role": "user"}],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "get_temperature",
                        "parameters_json_schema": {
                            "properties": {"city": {"type": "STRING"}},
                            "type": "OBJECT",
                        },
                        "response_json_schema": {"type": "STRING"},
                    }
                ]
            }
        ],
        "generationConfig": generation_config,
    }

    normalized = json.loads(vcr_request_body_normalizer(json.dumps(request)))

    assert normalized["contents"] == request["contents"]
    assert normalized["generationConfig"] == generation_config
    assert normalized["tools"][0]["functionDeclarations"] == [
        {
            "name": "get_temperature",
            "parameters": {
                "properties": {"city": {"type": "string"}},
                "type": "object",
            },
        }
    ]
