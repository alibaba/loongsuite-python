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

from typing import Optional

from google.genai import types

from opentelemetry.instrumentation.google_genai.message import (
    StreamOutputMessageAccumulator,
)
from opentelemetry.util.genai.types import Reasoning, Text, ToolCall


def _candidate(
    index: int,
    *parts: types.Part,
    finish_reason: Optional[types.FinishReason] = None,
) -> types.Candidate:
    return types.Candidate(
        index=index,
        finish_reason=finish_reason,
        content=types.Content(role="model", parts=list(parts)),
    )


def test_groups_text_chunks_by_candidate_index() -> None:
    accumulator = StreamOutputMessageAccumulator()
    accumulator.add_candidates(
        [
            _candidate(0, types.Part(text="first")),
            _candidate(1, types.Part(text="second")),
        ]
    )
    accumulator.add_candidates(
        [
            _candidate(0, types.Part(text=" answer")),
            _candidate(1, types.Part(text=" choice")),
        ]
    )
    accumulator.add_candidates(
        [
            _candidate(
                0,
                types.Part(text=""),
                finish_reason=types.FinishReason.STOP,
            ),
            _candidate(
                1,
                types.Part(text=""),
                finish_reason=types.FinishReason.MAX_TOKENS,
            ),
        ]
    )

    messages = accumulator.to_output_messages()

    assert len(messages) == 2
    assert messages[0].parts == [Text(content="first answer")]
    assert messages[0].finish_reason == "stop"
    assert messages[1].parts == [Text(content="second choice")]
    assert messages[1].finish_reason == "length"


def test_preserves_reasoning_and_text_as_separate_parts() -> None:
    accumulator = StreamOutputMessageAccumulator()
    accumulator.add_candidates(
        [_candidate(0, types.Part(text="think ", thought=True))]
    )
    accumulator.add_candidates(
        [_candidate(0, types.Part(text="carefully", thought=True))]
    )
    accumulator.add_candidates([_candidate(0, types.Part(text="answer"))])
    accumulator.add_candidates(
        [
            _candidate(
                0,
                types.Part(text=""),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )

    message = accumulator.to_output_messages()[0]

    assert message.parts == [
        Reasoning(content="think carefully"),
        Text(content="answer"),
    ]
    assert message.finish_reason == "stop"


def test_merges_function_call_fragments() -> None:
    accumulator = StreamOutputMessageAccumulator()
    accumulator.add_candidates(
        [
            _candidate(
                0,
                types.Part.from_function_call(
                    name="get_weather",
                    args={"city": "Hangzhou"},
                ),
            )
        ]
    )
    accumulator.add_candidates(
        [
            _candidate(
                0,
                types.Part.from_function_call(
                    name="get_weather",
                    args={"unit": "C"},
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )

    message = accumulator.to_output_messages()[0]

    assert len(message.parts) == 1
    assert isinstance(message.parts[0], ToolCall)
    assert message.parts[0].name == "get_weather"
    assert message.parts[0].arguments == {
        "city": "Hangzhou",
        "unit": "C",
    }
