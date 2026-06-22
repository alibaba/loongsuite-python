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

"""Unit tests for the DeerFlow instrumentation fixes.

These tests cover the regressions flagged by the code review:

* ``task_tool`` wrapper must propagate the original exception from ``wrapped``
  even when content capture is enabled (no ``UnboundLocalError``).
* ``SandboxProvider.acquire_async`` wrapper must return the awaited value
  (``str``), not a coroutine object.
* TASK span must be the parent of the langchain TOOL span emitted inside
  ``wrapped``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Make the plugin + util-genai importable without installation.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
_UTIL_GENAI_SRC = _REPO_ROOT / "util" / "opentelemetry-util-genai" / "src"
if _UTIL_GENAI_SRC.is_dir() and str(_UTIL_GENAI_SRC) not in sys.path:
    sys.path.insert(0, str(_UTIL_GENAI_SRC))

_PLUGIN_SRC = Path(__file__).resolve().parents[1] / "src"
if _PLUGIN_SRC.is_dir() and str(_PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_SRC))


from opentelemetry.instrumentation.deer_flow.patches.sandbox import (  # noqa: E402
    _ProviderAcquireAsyncWrapper,
)
from opentelemetry.instrumentation.deer_flow.patches.task_tool import (  # noqa: E402
    _TaskToolWrapper,
)
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode, get_tracer  # noqa: E402


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    yield provider
    provider.shutdown()


@pytest.fixture(autouse=True)
def _enable_capture(monkeypatch: pytest.MonkeyPatch):
    """Force ``_should_capture_content()`` to return True for all tests."""

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    monkeypatch.setenv(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )


def _make_tracer(tracer_provider: TracerProvider):
    return get_tracer(
        "opentelemetry.instrumentation.deer_flow",
        tracer_provider=tracer_provider,
    )


def test_task_tool_fails_when_capture_content_enabled(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    """Blocker 1: ``result = None`` must be initialized before ``try``.

    With capture enabled and ``wrapped`` raising, the original exception must
    propagate — not be swallowed by an ``UnboundLocalError`` on ``result``.
    """

    class _Boom(Exception):
        pass

    async def _raising_wrapped(*a: Any, **k: Any) -> str:
        raise _Boom("wrapped failed")

    _raising_wrapped.__name__ = "task_tool"

    wrapper = _TaskToolWrapper(handler=None)
    wrapper._tracer = _make_tracer(tracer_provider)

    with pytest.raises(_Boom, match="wrapped failed"):
        asyncio.run(
            wrapper.__call__(
                _raising_wrapped,
                None,
                (None, "describe", "do thing", "researcher", "tool_call_id_1"),
                {},
            )
        )

    spans = span_exporter.get_finished_spans()
    assert spans, "expected at least one span"
    task_span = spans[0]
    assert task_span.name == "run_task subagent.invoke:researcher"
    assert task_span.status.status_code == StatusCode.ERROR
    exc_events = [e for e in task_span.events if e.name == "exception"]
    assert exc_events, "expected an exception event on the TASK span"
    exc_attrs = exc_events[0].attributes or {}
    assert exc_attrs.get("exception.type", "").endswith("_Boom"), exc_attrs.get(
        "exception.type"
    )
    assert "UnboundLocalError" not in str(
        exc_attrs.get("exception.message", "")
    )


def test_acquire_async_returns_value_not_coroutine(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    """Blocker 2: ``acquire_async`` wrapper must await the coroutine.

    Previously the wrapper ran ``self._run`` in ``asyncio.to_thread`` and
    called ``wrapped(*args, **kwargs)`` synchronously, returning a coroutine
    object instead of the awaited ``str`` sandbox id.
    """

    async def _acquire_async(*a: Any, **k: Any) -> str:
        return "sid"

    _acquire_async.__name__ = "acquire_async"

    wrapper = _ProviderAcquireAsyncWrapper(_make_tracer(tracer_provider))

    result = asyncio.run(wrapper.__call__(_acquire_async, None, ("tid-123",), {}))

    assert result == "sid"
    assert not asyncio.iscoroutine(result), (
        f"acquire_async returned a coroutine: {result!r}"
    )

    spans = span_exporter.get_finished_spans()
    sandbox_spans = [
        s for s in spans if s.name == "run_task sandbox.acquire_async"
    ]
    assert sandbox_spans, [s.name for s in spans]
    span = sandbox_spans[0]
    assert span.end_time > span.start_time


def test_task_span_is_parent_of_langchain_tool_span(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    """Non-Blocker 3: TASK span must attach context so child TOOL spans
    parent to it.

    We simulate the langchain TOOL span by starting a span inside the
    ``wrapped`` body via the same tracer provider. If the TASK span is
    attached to the context, the inner span's ``parent_span_id`` equals the
    TASK span id.
    """

    inner_tracer = _make_tracer(tracer_provider)

    async def _wrapped_with_inner_tool(*a: Any, **k: Any) -> str:
        with inner_tracer.start_as_current_span("execute_tool task") as inner:
            inner.set_attribute("gen_ai.tool.name", "task")
            return "ok"

    _wrapped_with_inner_tool.__name__ = "task_tool"

    wrapper = _TaskToolWrapper(handler=None)
    wrapper._tracer = inner_tracer

    asyncio.run(
        wrapper.__call__(
            _wrapped_with_inner_tool,
            None,
            (None, "describe", "do thing", "researcher", "tool_call_id_1"),
            {},
        )
    )

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    assert "run_task subagent.invoke:researcher" in spans, list(spans)
    task_span = spans["run_task subagent.invoke:researcher"]
    tool_span = spans.get("execute_tool task")
    assert tool_span is not None, list(spans)
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == task_span.context.span_id, (
        "TOOL span parent is not the TASK span — context attach missing"
    )
