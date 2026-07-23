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

from google.genai import types

from opentelemetry.instrumentation.google_genai._utils import (
    ResponseAccumulator,
    apply_response,
    create_llm_invocation,
)
from opentelemetry.util.genai.types import (
    Blob,
    Reasoning,
    Text,
    ToolCall,
    Uri,
)


def _response(text, *, finish_reason=None, response_id=None):
    return types.GenerateContentResponse(
        response_id=response_id,
        model_version="gemini-2.5-flash-001",
        candidates=[
            types.Candidate(
                index=0,
                finish_reason=finish_reason,
                content=types.Content(
                    role="model", parts=[types.Part(text=text)]
                ),
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=3,
            candidates_token_count=5,
            cached_content_token_count=2,
        ),
    )


def test_create_invocation_maps_request_messages_config_and_tools():
    config = types.GenerateContentConfig(
        system_instruction="Be concise",
        temperature=0.2,
        top_p=0.8,
        top_k=20,
        max_output_tokens=128,
        candidate_count=2,
        response_mime_type="application/json",
        tools=[
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="weather",
                        description="Look up weather",
                        parameters={
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    )
                ]
            )
        ],
    )
    invocation = create_llm_invocation(
        SimpleNamespace(vertexai=False),
        model="gemini-2.5-flash",
        contents="hello",
        config=config,
        function_name="google.genai.Models.generate_content",
        extra_attributes={"custom": "value"},
    )

    assert invocation.operation_name == "generate_content"
    assert invocation.provider == "gcp.gen_ai"
    assert invocation.input_messages[0].role == "user"
    assert invocation.input_messages[0].parts == [Text(content="hello")]
    assert invocation.system_instruction == [Text(content="Be concise")]
    assert invocation.tool_definitions[0].name == "weather"
    assert invocation.temperature == 0.2
    assert invocation.top_p == 0.8
    assert invocation.top_k == 20
    assert invocation.max_tokens == 128
    assert invocation.choice_count == 2
    assert invocation.output_type == "json"
    assert invocation.attributes["custom"] == "value"


def test_create_invocation_maps_multimodal_reasoning_and_tool_parts():
    contents = types.Content(
        role="model",
        parts=[
            types.Part(text="thinking", thought=True),
            types.Part(
                inline_data=types.Blob(mime_type="image/png", data=b"png")
            ),
            types.Part(
                file_data=types.FileData(
                    mime_type="audio/wav", file_uri="gs://bucket/audio.wav"
                )
            ),
            types.Part(
                function_call=types.FunctionCall(
                    id="call-1", name="weather", args={"city": "Paris"}
                )
            ),
        ],
    )
    invocation = create_llm_invocation(
        SimpleNamespace(vertexai=True),
        model="gemini-2.5-flash",
        contents=contents,
        config=None,
        function_name="google.genai.Models.generate_content",
    )

    parts = invocation.input_messages[0].parts
    assert invocation.provider == "gcp.vertex_ai"
    assert isinstance(parts[0], Reasoning)
    assert isinstance(parts[1], Blob)
    assert isinstance(parts[2], Uri)
    assert isinstance(parts[3], ToolCall)


def test_apply_response_preserves_provider_identity_and_usage():
    invocation = create_llm_invocation(
        SimpleNamespace(vertexai=False),
        model="gemini-2.5-flash",
        contents="hello",
        config=None,
        function_name="google.genai.Models.generate_content",
    )
    apply_response(
        invocation,
        _response(
            "world",
            finish_reason=types.FinishReason.STOP,
            response_id="response-123",
        ),
    )

    assert invocation.response_id == "response-123"
    assert invocation.response_model_name == "gemini-2.5-flash-001"
    assert invocation.input_tokens == 3
    assert invocation.output_tokens == 5
    assert invocation.usage_cache_read_input_tokens == 2
    assert invocation.finish_reasons == ["stop"]
    assert invocation.output_messages[0].parts == [Text(content="world")]


def test_apply_response_maps_unspecified_finish_reason_to_error():
    invocation = create_llm_invocation(
        SimpleNamespace(vertexai=False),
        model="gemini-2.5-flash",
        contents="hello",
        config=None,
        function_name="google.genai.Models.generate_content",
    )

    apply_response(
        invocation,
        _response(
            "",
            finish_reason=types.FinishReason.FINISH_REASON_UNSPECIFIED,
        ),
    )

    assert invocation.finish_reasons == ["error"]


def test_response_accumulator_merges_stream_deltas():
    accumulator = ResponseAccumulator()
    accumulator.add(_response("he"))
    accumulator.add(
        _response("llo", finish_reason=types.FinishReason.MAX_TOKENS)
    )

    messages = accumulator.output_messages()
    assert len(messages) == 1
    assert messages[0].parts == [Text(content="hello")]
    assert messages[0].finish_reason == "length"
