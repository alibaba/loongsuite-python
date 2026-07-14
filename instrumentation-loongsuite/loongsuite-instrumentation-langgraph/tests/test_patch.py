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

"""Tests that ``create_react_agent`` and ``Pregel.stream`` are correctly
patched by the LangGraphInstrumentor.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from opentelemetry.instrumentation.langgraph import LangGraphInstrumentor
from opentelemetry.instrumentation.langgraph.internal.patch import (
    AGENT_FLAVOR_METADATA_KEY,
    LANGGRAPH_PREBUILT_AGENT_FLAVOR,
    _get_graph_agent_flavor,
    _inject_agent_metadata,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeChatModelWithTools(FakeListChatModel):
    """``FakeListChatModel`` extended with ``bind_tools`` so that
    ``create_react_agent`` can call ``model.bind_tools(tools)``.
    """

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> Any:
        return self


@tool
def _dummy_tool(query: str) -> str:
    """A no-op tool used only for testing."""
    return f"result for {query}"


def _build_react_agent(**extra_kwargs: Any):
    """Create a simple ReAct agent via ``create_react_agent``."""
    # Import inside function so each call picks up the current (possibly
    # patched) module-level attribute rather than a stale top-level reference.
    from langgraph.prebuilt import create_react_agent  # noqa: PLC0415

    llm = _FakeChatModelWithTools(responses=["Hello!"])
    return create_react_agent(llm, [_dummy_tool], **extra_kwargs)


# ---------------------------------------------------------------------------
# Tests: create_react_agent patching
# ---------------------------------------------------------------------------


class TestCreateReactAgentPatch:
    """Verify that instrumenting patches ``create_react_agent``."""

    def test_patch_adds_react_flag(self, instrument):
        """The compiled graph has ``_loongsuite_react_agent = True``."""
        graph = _build_react_agent()
        assert getattr(graph, "_loongsuite_react_agent", False) is True
        assert (
            getattr(graph, AGENT_FLAVOR_METADATA_KEY)
            == LANGGRAPH_PREBUILT_AGENT_FLAVOR
        )

    def test_patch_does_not_override_default_name(self, instrument):
        """The original graph name is preserved when no explicit name
        is given.
        """
        graph = _build_react_agent()
        assert graph.name != ""
        assert getattr(graph, "_loongsuite_react_agent", False) is True

    def test_patch_preserves_explicit_name(self, instrument):
        """When the caller provides an explicit *name*, it is preserved."""
        graph = _build_react_agent(name="my_custom_agent")
        assert graph.name == "my_custom_agent"
        assert getattr(graph, "_loongsuite_react_agent", False) is True
        assert (
            getattr(graph, AGENT_FLAVOR_METADATA_KEY)
            == LANGGRAPH_PREBUILT_AGENT_FLAVOR
        )

    def test_uninstrument_restores_original(self):
        """After ``uninstrument()``, ``create_react_agent`` returns an
        unpatched graph (no ``_loongsuite_react_agent`` attribute).
        """
        instrumentor = LangGraphInstrumentor()
        instrumentor.instrument()
        instrumentor.uninstrument()

        graph = _build_react_agent()
        assert not getattr(graph, "_loongsuite_react_agent", False)
        assert not hasattr(graph, AGENT_FLAVOR_METADATA_KEY)

    def test_double_instrument_is_safe(self):
        """Calling ``instrument()`` twice does not break the wrapper."""
        instrumentor = LangGraphInstrumentor()
        instrumentor.instrument()
        try:
            instrumentor.instrument()
            graph = _build_react_agent()
            assert getattr(graph, "_loongsuite_react_agent", False) is True
        finally:
            instrumentor.uninstrument()


# ---------------------------------------------------------------------------
# Tests: Pregel.stream metadata injection
# ---------------------------------------------------------------------------


class TestPregelStreamPatch:
    """Verify that ``Pregel.stream`` injects metadata for ReAct agents."""

    def test_metadata_injected_for_react_agent(
        self, instrument, span_exporter
    ):
        """Invoking a ReAct agent should inject metadata into the config.

        Metadata injection is verified indirectly: when the metadata flag is
        present, LangChain's LoongsuiteTracer creates Agent/ReAct Step spans
        instead of generic chain spans. Absence of Agent span would indicate
        the Pregel wrapper stopped injecting metadata.
        """
        graph = _build_react_agent()
        result = graph.invoke({"messages": [("user", "hello")]})
        assert result is not None

        spans = span_exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]
        assert len(agent_spans) == 1, (
            f"Expected 1 Agent span (metadata injected), got {len(agent_spans)}: "
            f"{[s.name for s in spans]}"
        )

    def test_no_metadata_for_plain_graph(self, instrument):
        """A plain (non-ReAct) graph should NOT have the metadata flag."""
        from langgraph.graph import StateGraph  # noqa: PLC0415

        class _State(dict):
            pass

        builder = StateGraph(_State)
        builder.add_node("node", lambda s: s)
        builder.set_entry_point("node")
        builder.set_finish_point("node")
        plain_graph = builder.compile()

        assert not getattr(plain_graph, "_loongsuite_react_agent", False)

    def test_metadata_injection_is_copy_on_write(self):
        callback = object()
        original = {
            "callbacks": [callback],
            "metadata": {"customer": "kept"},
        }

        rewritten = _inject_agent_metadata(
            original, LANGGRAPH_PREBUILT_AGENT_FLAVOR
        )

        assert rewritten is not original
        assert rewritten["metadata"] is not original["metadata"]
        assert original == {
            "callbacks": [callback],
            "metadata": {"customer": "kept"},
        }
        assert rewritten["callbacks"] == [callback]
        assert rewritten["metadata"]["customer"] == "kept"
        assert (
            rewritten["metadata"][AGENT_FLAVOR_METADATA_KEY]
            == LANGGRAPH_PREBUILT_AGENT_FLAVOR
        )

    def test_explicit_flavor_does_not_require_legacy_marker(self):
        graph = type("Graph", (), {})()
        setattr(graph, AGENT_FLAVOR_METADATA_KEY, "deerflow")

        assert _get_graph_agent_flavor(graph) == "deerflow"

    def test_uninstrument_restores_stream(self):
        """After uninstrument, Pregel.stream is no longer a wrapt wrapper."""
        from langgraph.pregel import Pregel  # noqa: PLC0415
        from wrapt import ObjectProxy  # noqa: PLC0415

        instrumentor = LangGraphInstrumentor()
        instrumentor.instrument()
        assert isinstance(Pregel.stream, ObjectProxy)

        instrumentor.uninstrument()
        assert not isinstance(Pregel.stream, ObjectProxy)


class TestCreateReactAgentIntegration:
    """Verify the patched graph still works correctly."""

    def test_agent_invoke_runs_without_error(self, instrument):
        """The patched agent can be invoked without errors."""
        graph = _build_react_agent()
        result = graph.invoke({"messages": [("user", "hello")]})

        assert result is not None
        messages = result.get("messages", [])
        assert len(messages) >= 1
        last_msg = messages[-1]
        assert isinstance(last_msg, AIMessage)
