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

"""Review follow-ups for the LLM layer.

Three gaps the review found are pinned down here: an async stream that is
awaited and then never read, a call whose usage only reaches ``LLMOutput.raw``,
and the chunk text of a stream that carries empty chunks. The fourth covers the
``LayerSpan.attach`` rollback: a span whose attach fails must not stay attached
to the caller's context, or the next call of the same trace is filed under a
span that never ends.

Every stub is driven through the real ``@trace_llm`` seam, so the behaviour
under test is the framework's own wrapper resolution rather than a stand-in for
it. The stubs live here instead of in ``conftest`` because only these tests need
them.
"""

import asyncio
from typing import Any, AsyncIterator, Dict

from agentuniverse.base.annotation.trace import trace_llm
from agentuniverse.llm.llm_output import LLMOutput, TokenUsage

from opentelemetry.instrumentation.agentuniverse import _common as common

from .conftest import StubLLM, StubTool
from .test_instrumentor import (
    LLM_SPAN,
    TOOL_SPAN,
    capture_on,
    layer_records,
    one_layer_record,
)


class AsyncStreamingLLM(StubLLM):
    """A ``@trace_llm`` coroutine that returns an async iterator.

    ``trace_llm`` sends a coroutine function to the async wrapper, and a result
    that is not an ``LLMOutput`` is what the wrapper treats as a stream, so this
    is the shape of a real streaming ``acall``.
    """

    name = "async_stream_llm"

    @trace_llm
    async def call(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[LLMOutput]:
        async def stream() -> AsyncIterator[LLMOutput]:
            yield LLMOutput(text="part-1")
            yield LLMOutput(
                text="part-2", usage=TokenUsage(text_in=4, text_out=6)
            )

        return stream()


class RawUsageLLM(StubLLM):
    """A call whose usage only arrives in the provider payload.

    ``OpenAIStyleLLM`` keeps the provider dict in ``LLMOutput.raw`` and fills
    ``LLMOutput.usage`` from it; a service that reports usage where the
    framework does not look leaves ``usage`` unset while ``raw`` still carries
    the numbers.
    """

    name = "raw_usage_llm"

    @trace_llm
    def call(self, prompt: str, **kwargs: Any) -> LLMOutput:
        return LLMOutput(
            text=f"raw:{prompt}",
            raw={
                "id": "resp-1",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
            },
            finish_reason="stop",
        )


class EmptyTextChunkLLM(StubLLM):
    """A stream whose empty chunks sit between the chunks that carry text.

    A role-only or usage-only chunk is a real framework payload with
    ``text=""``; it contributes nothing, and certainly not its own repr.
    """

    name = "empty_text_chunk_llm"

    @trace_llm
    def call(self, prompt: str, **kwargs: Any):
        yield LLMOutput(text="hello ")
        yield LLMOutput(text="")
        yield LLMOutput(text="world", usage=TokenUsage(text_in=1, text_out=2))


class AsyncEmptyTextChunkLLM(EmptyTextChunkLLM):
    """``EmptyTextChunkLLM`` over the async wrapper."""

    name = "async_empty_text_chunk_llm"

    @trace_llm
    async def call(self, prompt: str, **kwargs: Any):
        async def stream() -> AsyncIterator[LLMOutput]:
            yield LLMOutput(text="hello ")
            yield LLMOutput(text="")
            yield LLMOutput(
                text="world", usage=TokenUsage(text_in=1, text_out=2)
            )

        return stream()


def span_named(harness: Any, prefix: str) -> Any:
    """The one exported span whose name starts with ``prefix``."""
    spans = [s for s in harness.spans() if s.name.startswith(prefix)]
    assert len(spans) == 1, [s.name for s in harness.spans()]
    return spans[0]


class TestAsyncStreamLifecycle:
    def test_async_stream_releases_the_context_before_it_is_read(
        self, loongsuite_harness: Any
    ) -> None:
        """The caller resumes outside the LLM span, so a later call is a root."""
        stream = None

        async def drive() -> Any:
            nonlocal stream
            stream = await AsyncStreamingLLM().call(prompt="hello")
            # Before reading anything the caller does other work; it must not
            # be filed under the still-open LLM span.
            StubTool().run(query="hello")
            await stream.aclose()

        asyncio.run(drive())
        llm = span_named(loongsuite_harness, LLM_SPAN)
        tool = span_named(loongsuite_harness, TOOL_SPAN)
        assert tool.parent is None
        assert tool.parent != llm.context

    def test_async_stream_closed_without_being_read_finalizes_once(
        self, loongsuite_harness: Any
    ) -> None:
        """A dropped stream still closes its span, with the empty output."""

        async def drive() -> None:
            stream = await AsyncStreamingLLM().call(prompt="hello")
            await stream.aclose()

        asyncio.run(drive())
        records = layer_records(loongsuite_harness, LLM_SPAN)
        assert len(records) == 1, [r.name for r in records]
        assert records[0].au["au.llm.streaming"] is True
        assert records[0].au["au.llm.status"] == "success"

    def test_async_stream_read_fully_finalizes_once(
        self, loongsuite_harness: Any
    ) -> None:
        """The ordinary case still ends exactly one span with both chunks."""

        async def drive() -> None:
            stream = await AsyncStreamingLLM().call(prompt="hello")
            assert [chunk.text for chunk in [c async for c in stream]] == [
                "part-1",
                "part-2",
            ]

        with capture_on():
            asyncio.run(drive())
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.output"] == "part-1part-2"
        assert record.au["au.llm.usage.total_tokens"] == 10

    def test_async_stream_consumed_in_another_task_finalizes_once(
        self, loongsuite_harness: Any
    ) -> None:
        """A stream handed to another task closes the same single span."""

        async def drive() -> None:
            stream = await AsyncStreamingLLM().call(prompt="hello")

            async def consume() -> None:
                async for _ in stream:
                    pass

            await asyncio.create_task(consume())

        with capture_on():
            asyncio.run(drive())
        records = layer_records(loongsuite_harness, LLM_SPAN)
        assert len(records) == 1, [r.name for r in records]
        assert records[0].au["au.llm.output"] == "part-1part-2"
        assert records[0].au["au.llm.status"] == "success"
        assert records[0].au["au.llm.usage.total_tokens"] == 10


class TestLayerSpanAttachRollback:
    def test_attach_failure_leaves_no_attached_span(
        self, loongsuite_harness: Any, monkeypatch: Any
    ) -> None:
        """A span whose attach fails is closed and detached before the raise."""
        original = common.reset_token_usage
        failing: Dict[str, bool] = {"on": True}

        def flaky(span_id: Any = None) -> None:
            if failing["on"]:
                raise RuntimeError("token bookkeeping exploded")
            original(span_id)

        monkeypatch.setattr(common, "reset_token_usage", flaky)
        StubLLM().call(prompt="first")
        failing["on"] = False
        StubLLM().call(prompt="second")

        spans = [
            s
            for s in loongsuite_harness.spans()
            if s.name.startswith(LLM_SPAN)
        ]
        complete = [
            s for s in spans if s.attributes.get("au.llm.status") == "success"
        ]
        assert len(complete) == 1, [s.name for s in spans]
        # The failed attach must have detached its span, or this one would be
        # its child.
        assert complete[0].parent is None


class TestUsageAndChunkText:
    def test_usage_from_the_raw_payload_reaches_both_namespaces(
        self, loongsuite_harness: Any
    ) -> None:
        RawUsageLLM().call(prompt="hello")
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.usage.total_tokens"] == 25
        assert record.au["au.llm.usage.prompt_tokens"] == 20
        assert record.au["au.llm.usage.completion_tokens"] == 5
        assert record.gen_ai.get("gen_ai.usage.input_tokens") == 20
        assert record.gen_ai.get("gen_ai.usage.output_tokens") == 5

    def test_usage_of_reads_the_real_adapter_output(
        self, loongsuite_harness: Any
    ) -> None:
        """The real adapter's construction path, not a hand-built TokenUsage."""
        import httpx
        from agentuniverse.llm.openai_style_llm import OpenAIStyleLLM
        from openai import OpenAI

        from opentelemetry.instrumentation.agentuniverse._llm import (
            usage_of,
        )

        completion = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            },
        }
        adapter = OpenAIStyleLLM(
            name="adapter_llm",
            model_name="synthetic-model",
            api_base="http://provider.invalid/v1",
        )
        with OpenAI(
            api_key="synthetic",
            base_url="http://provider.invalid/v1",
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=completion)
                )
            ),
        ) as client:
            adapter.client = client
            result = adapter.call(
                messages=[{"role": "user", "content": "hello"}]
            )
        assert result.raw["usage"]["total_tokens"] == 25
        assert usage_of(result).total_tokens == 25

    def test_empty_text_chunks_contribute_nothing(
        self, loongsuite_harness: Any
    ) -> None:
        with capture_on():
            list(EmptyTextChunkLLM().call(prompt="hello"))
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.output"] == "hello world"

    def test_empty_text_chunks_contribute_nothing_async(
        self, loongsuite_harness: Any
    ) -> None:
        async def drive() -> None:
            stream = await AsyncEmptyTextChunkLLM().call(prompt="hello")
            async for _ in stream:
                pass

        with capture_on():
            asyncio.run(drive())
        record = one_layer_record(loongsuite_harness, LLM_SPAN)
        assert record.au["au.llm.output"] == "hello world"
