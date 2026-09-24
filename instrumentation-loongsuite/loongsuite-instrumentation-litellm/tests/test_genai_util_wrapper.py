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

import asyncio
import json
from types import SimpleNamespace

import litellm

from opentelemetry import context as otel_context
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor


def _chat_response(model: str, content: str):
    return SimpleNamespace(
        id=f"chatcmpl-{model}",
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=4,
            completion_tokens=3,
            total_tokens=7,
        ),
    )


def _embedding_response(model: str):
    return SimpleNamespace(
        id=f"embd-{model}",
        model=model,
        data=[{"embedding": [0.1, 0.2, 0.3]}],
        usage=SimpleNamespace(
            prompt_tokens=5,
            total_tokens=5,
        ),
    )


def _chunk(choices, usage=None):
    return SimpleNamespace(
        id="chatcmpl-stream",
        model="qwen-turbo",
        choices=choices,
        usage=usage,
    )


def _choice(
    index,
    content=None,
    finish_reason=None,
    tool_calls=None,
    reasoning_content=None,
):
    return SimpleNamespace(
        index=index,
        delta=SimpleNamespace(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        ),
        finish_reason=finish_reason,
    )


def _tool_delta(index, tool_call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=tool_call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _ClosableIterator:
    def __init__(self, chunks):
        self._iterator = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self):
        self.closed = True


class _AsyncClosableStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._index = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration

        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def aclose(self):
        self.closed = True


def test_completion_positional_args_feed_genai_invocation(
    monkeypatch, tracer_provider, span_exporter
):
    def fake_completion(model, messages, **kwargs):
        assert model == "qwen-turbo"
        assert messages[0]["content"] == "hello"
        assert kwargs["temperature"] == 0.2
        return _chat_response(model, "hello back")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        litellm.completion(
            "qwen-turbo",
            [{"role": "user", "content": "hello"}],
            temperature=0.2,
        )
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.span.kind"] == "LLM"
    assert span.attributes["gen_ai.provider.name"] == "dashscope"
    assert span.attributes["gen_ai.request.model"] == "qwen-turbo"

    input_messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == "hello"


def test_provider_prefers_custom_provider_over_model_heuristic_and_system_split(
    monkeypatch, tracer_provider, span_exporter
):
    def fake_completion(model, messages, **kwargs):
        assert model == "gpt-4"
        assert kwargs["custom_llm_provider"] == "azure"
        assert messages[0]["role"] == "system"
        return _chat_response(model, "azure response")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        litellm.completion(
            model="gpt-4",
            custom_llm_provider="azure",
            messages=[
                {"role": "system", "content": "system rules"},
                {"role": "developer", "content": "developer rules"},
                {"role": "user", "content": "hello"},
            ],
        )
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.provider.name"] == "azure"

    input_messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert [message["role"] for message in input_messages] == ["user"]

    system_instructions = json.loads(
        span.attributes["gen_ai.system_instructions"]
    )
    assert [part["content"] for part in system_instructions] == [
        "system rules",
        "developer rules",
    ]


def test_provider_prefers_known_base_url_over_custom_adapter(
    monkeypatch, tracer_provider, span_exporter
):
    def fake_completion(model, messages, **kwargs):
        assert model == "custom-compatible-model"
        assert kwargs["custom_llm_provider"] == "openai"
        assert (
            kwargs["api_base"]
            == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        return _chat_response(model, "compatible response")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        litellm.completion(
            model="custom-compatible-model",
            custom_llm_provider="openai",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            messages=[{"role": "user", "content": "hello"}],
        )
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.provider.name"] == "dashscope"


def test_completion_usage_falls_back_to_total_minus_prompt_tokens(
    monkeypatch, tracer_provider, span_exporter
):
    def fake_completion(model, messages, **kwargs):
        assert model == "qwen-turbo"
        return SimpleNamespace(
            id="chatcmpl-fallback-usage",
            model=model,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="fallback usage",
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=4, total_tokens=9),
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        litellm.completion(
            model="qwen-turbo",
            messages=[{"role": "user", "content": "hello"}],
        )
    finally:
        instrumentor.uninstrument()

    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.usage.input_tokens"] == 4
    assert span.attributes["gen_ai.usage.output_tokens"] == 5
    assert span.attributes["gen_ai.usage.total_tokens"] == 9


def test_suppressed_instrumentation_skips_completion_span(
    monkeypatch, tracer_provider, span_exporter
):
    def fake_completion(model, messages, **kwargs):
        return _chat_response(model, "not traced")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    instrumentor = LiteLLMInstrumentor()
    token = None
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        ctx = otel_context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True)
        token = otel_context.attach(ctx)
        litellm.completion(
            model="qwen-turbo",
            messages=[{"role": "user", "content": "hello"}],
        )
    finally:
        if token is not None:
            otel_context.detach(token)
        instrumentor.uninstrument()

    assert not span_exporter.get_finished_spans()


def test_no_content_mode_omits_messages_but_keeps_metadata(
    monkeypatch, tracer_provider, span_exporter
):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "NO_CONTENT"
    )

    def fake_completion(model, messages, **kwargs):
        return _chat_response(model, "content hidden")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        litellm.completion(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": "secret system"},
                {"role": "user", "content": "secret user"},
            ],
        )
    finally:
        instrumentor.uninstrument()

    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.span.kind"] == "LLM"
    assert span.attributes["gen_ai.request.model"] == "qwen-turbo"
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.output.messages" not in span.attributes
    assert "gen_ai.system_instructions" not in span.attributes


def test_embedding_usage_records_input_tokens_only(
    monkeypatch, tracer_provider, span_exporter
):
    def fake_embedding(model, input_, **kwargs):
        assert model == "text-embedding-v1"
        assert input_ == "embed me"
        return _embedding_response(model)

    monkeypatch.setattr(litellm, "embedding", fake_embedding)

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        litellm.embedding(
            "text-embedding-v1",
            "embed me",
            custom_llm_provider="openai",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.span.kind"] == "EMBEDDING"
    assert span.attributes["gen_ai.provider.name"] == "dashscope"
    assert span.attributes["gen_ai.usage.input_tokens"] == 5
    assert span.attributes["gen_ai.usage.total_tokens"] == 5
    assert "gen_ai.usage.output_tokens" not in span.attributes
    assert span.attributes["gen_ai.embeddings.dimension.count"] == 3


def test_streaming_completion_records_ttft_choices_and_tool_calls(
    monkeypatch, tracer_provider, span_exporter
):
    chunks = [
        _chunk(
            [
                # Keep this split to avoid a codespell false positive.
                _choice(0, content="he"),
                _choice(1, content="bon"),
            ]
        ),
        _chunk(
            [
                _choice(
                    0,
                    content="llo",
                    tool_calls=[
                        _tool_delta(
                            0,
                            tool_call_id="call_1",
                            name="lookup",
                            arguments='{"q":',
                        )
                    ],
                ),
                _choice(1, content="jour"),
            ]
        ),
        _chunk(
            [
                _choice(
                    0,
                    tool_calls=[_tool_delta(0, arguments={"ignored": True})],
                ),
            ]
        ),
        _chunk(
            [
                _choice(
                    0,
                    finish_reason="tool_calls",
                    tool_calls=[_tool_delta(0, arguments='"weather"}')],
                ),
                _choice(1, finish_reason="stop"),
            ],
            usage=SimpleNamespace(
                prompt_tokens=6,
                completion_tokens=5,
                total_tokens=11,
            ),
        ),
    ]

    def fake_completion(*args, **kwargs):
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}
        return iter(chunks)

    monkeypatch.setattr(litellm, "completion", fake_completion)

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        response = litellm.completion(
            model="qwen-turbo",
            messages=[{"role": "user", "content": "stream please"}],
            stream=True,
            n=2,
        )
        assert len(list(response)) == 4
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert "gen_ai.response.time_to_first_token" in span.attributes
    assert span.attributes["gen_ai.request.choice.count"] == 2
    assert span.attributes["gen_ai.usage.input_tokens"] == 6
    assert span.attributes["gen_ai.usage.output_tokens"] == 5

    output_messages = json.loads(span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 2
    assert output_messages[0]["parts"][0]["content"] == "hello"
    assert output_messages[1]["parts"][0]["content"] == "bonjour"
    tool_call = output_messages[0]["parts"][1]
    assert tool_call["type"] == "tool_call"
    assert tool_call["id"] == "call_1"
    assert tool_call["name"] == "lookup"
    assert tool_call["arguments"] == {"q": "weather"}


def test_streaming_reasoning_multimodal_content_and_empty_choice(
    monkeypatch, tracer_provider, span_exporter
):
    chunks = [
        _chunk(
            [
                _choice(0, reasoning_content="thinking"),
                _choice(1, finish_reason="stop"),
            ]
        ),
        _chunk([_choice(0, content={"unexpected": True})]),
        _chunk(
            [
                _choice(
                    0,
                    content=[
                        {"type": "text", "text": "hello"},
                        {"type": "image_url", "image_url": {"url": "x"}},
                        " world",
                    ],
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
            ),
        ),
    ]

    def fake_completion(*args, **kwargs):
        assert kwargs["stream"] is True
        return iter(chunks)

    monkeypatch.setattr(litellm, "completion", fake_completion)

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        response = litellm.completion(
            model="qwen-turbo",
            messages=[{"role": "user", "content": "reason"}],
            stream=True,
            n=2,
        )
        assert len(list(response)) == 3
    finally:
        instrumentor.uninstrument()

    span = span_exporter.get_finished_spans()[0]
    assert "gen_ai.response.time_to_first_token" in span.attributes
    output_messages = json.loads(span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 2
    assert output_messages[0]["parts"][0] == {
        "content": "thinking",
        "type": "reasoning",
    }
    assert output_messages[0]["parts"][1] == {
        "content": "hello world",
        "type": "text",
    }
    assert output_messages[1]["parts"] == [{"content": "", "type": "text"}]


def test_streaming_close_closes_underlying_stream_and_finalizes(
    monkeypatch, tracer_provider, span_exporter
):
    stream = _ClosableIterator(
        [
            _chunk([_choice(0, content="partial")]),
            _chunk([_choice(0, content=" ignored", finish_reason="stop")]),
        ]
    )

    def fake_completion(*args, **kwargs):
        assert kwargs["stream"] is True
        return stream

    monkeypatch.setattr(litellm, "completion", fake_completion)

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        response = litellm.completion(
            model="qwen-turbo",
            messages=[{"role": "user", "content": "stream"}],
            stream=True,
        )
        next(response)
        response.close()
    finally:
        instrumentor.uninstrument()

    assert stream.closed is True
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    output_messages = json.loads(spans[0].attributes["gen_ai.output.messages"])
    assert output_messages[0]["parts"][0]["content"] == "partial"


def test_async_streaming_aclose_closes_stream_and_finalizes(
    monkeypatch, tracer_provider, span_exporter
):
    captured = {}

    async def fake_acompletion(*args, **kwargs):
        assert kwargs["stream"] is True
        stream = _AsyncClosableStream(
            [
                _chunk([_choice(0, content="async partial")]),
                _chunk([_choice(0, content=" ignored", finish_reason="stop")]),
            ]
        )
        captured["stream"] = stream
        return stream

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    async def run_call():
        response = await litellm.acompletion(
            model="qwen-turbo",
            messages=[{"role": "user", "content": "stream"}],
            stream=True,
        )
        iterator = response.__aiter__()
        await iterator.__anext__()
        await iterator.aclose()

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        asyncio.run(run_call())
    finally:
        instrumentor.uninstrument()

    assert captured["stream"].closed is True
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    output_messages = json.loads(spans[0].attributes["gen_ai.output.messages"])
    assert output_messages[0]["parts"][0]["content"] == "async partial"


def test_async_completion_concurrent_calls_keep_separate_spans(
    monkeypatch, tracer_provider, span_exporter
):
    async def fake_acompletion(model, messages, **kwargs):
        await asyncio.sleep(0.01 if model == "qwen-turbo" else 0)
        return _chat_response(model, f"reply to {messages[0]['content']}")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    async def run_calls():
        return await asyncio.gather(
            litellm.acompletion(
                "qwen-turbo",
                [{"role": "user", "content": "first"}],
            ),
            litellm.acompletion(
                "qwen-plus",
                [{"role": "user", "content": "second"}],
            ),
        )

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        asyncio.run(run_calls())
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    observed = {
        json.loads(span.attributes["gen_ai.input.messages"])[0]["parts"][0][
            "content"
        ]: span.attributes["gen_ai.request.model"]
        for span in spans
    }
    assert observed == {"first": "qwen-turbo", "second": "qwen-plus"}
