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

"""Unit tests for opentelemetry.instrumentation.openhands.internal.session_context."""

from __future__ import annotations

from opentelemetry import context as otel_context
from opentelemetry.instrumentation.openhands.internal.session_context import (
    AttachedSession,
    clear_all,
    clear_context,
    get_context,
    get_tool_definition,
    get_tool_registry,
    store_context,
    store_tool_registry,
)


def setup_function():
    clear_all()


def teardown_function():
    clear_all()


# ---------------------------------------------------------------------------
# store_context / get_context / clear_context
# ---------------------------------------------------------------------------


def test_store_and_get_context():
    ctx = otel_context.get_current()
    store_context("s1", ctx)
    assert get_context("s1") is ctx


def test_store_empty_sid():
    ctx = otel_context.get_current()
    store_context("", ctx)
    store_context(None, ctx)
    # Should not store anything
    assert get_context("") is None


def test_get_context_fallback_to_last_sid():
    ctx = otel_context.get_current()
    store_context("last", ctx)
    # Querying unknown sid falls back to _last_sid
    assert get_context("unknown") is ctx


def test_get_context_none_sid_falls_back():
    ctx = otel_context.get_current()
    store_context("x", ctx)
    assert get_context(None) is ctx


def test_get_context_no_stored():
    assert get_context("nothing") is None


def test_clear_context():
    ctx = otel_context.get_current()
    store_context("s1", ctx)
    clear_context("s1")
    assert get_context("s1") is None


def test_clear_context_empty_sid():
    # Should be a no-op
    clear_context("")
    clear_context(None)


def test_clear_context_resets_last_sid():
    ctx = otel_context.get_current()
    store_context("s1", ctx)
    clear_context("s1")
    # _last_sid was "s1", now cleared
    assert get_context(None) is None


def test_clear_all():
    ctx = otel_context.get_current()
    store_context("a", ctx)
    store_context("b", ctx)
    store_tool_registry("a", [{"type": "function", "function": {"name": "t"}}])
    clear_all()
    assert get_context("a") is None
    assert get_context("b") is None
    assert get_tool_registry("a") is None


# ---------------------------------------------------------------------------
# store_tool_registry / get_tool_definition / get_tool_registry
# ---------------------------------------------------------------------------


def test_store_and_get_tool_registry():
    tools = [
        {
            "type": "function",
            "function": {"name": "execute_bash", "description": "Run bash"},
        },
        {
            "type": "function",
            "function": {"name": "file_read", "description": "Read file"},
        },
    ]
    store_tool_registry("s1", tools)
    reg = get_tool_registry("s1")
    assert reg is not None
    assert "execute_bash" in reg
    assert "file_read" in reg


def test_get_tool_definition_found():
    tools = [
        {
            "type": "function",
            "function": {"name": "execute_bash", "description": "Run bash"},
        },
    ]
    store_tool_registry("s1", tools)
    td = get_tool_definition("s1", "execute_bash")
    assert td is not None
    assert td["function"]["name"] == "execute_bash"


def test_get_tool_definition_not_found():
    tools = [
        {"type": "function", "function": {"name": "execute_bash"}},
    ]
    store_tool_registry("s1", tools)
    assert get_tool_definition("s1", "nonexistent") is None


def test_get_tool_definition_no_name():
    assert get_tool_definition("s1", None) is None
    assert get_tool_definition("s1", "") is None


def test_get_tool_definition_fallback_to_last_sid():
    from opentelemetry import context as ctx

    tools = [
        {"type": "function", "function": {"name": "bash"}},
    ]
    # _last_sid is set by store_context, not store_tool_registry
    store_context("last-sid", ctx.get_current())
    store_tool_registry("last-sid", tools)
    # Query with unknown sid, falls back to _last_sid
    td = get_tool_definition("unknown-sid", "bash")
    assert td is not None


def test_get_tool_registry_fallback_to_last_sid():
    from opentelemetry import context as ctx

    tools = [
        {"type": "function", "function": {"name": "bash"}},
    ]
    store_context("last-sid", ctx.get_current())
    store_tool_registry("last-sid", tools)
    reg = get_tool_registry("unknown-sid")
    assert reg is not None
    assert "bash" in reg


def test_get_tool_registry_none():
    assert get_tool_registry("nothing") is None


def test_store_tool_registry_empty_sid():
    tools = [{"type": "function", "function": {"name": "t"}}]
    store_tool_registry("", tools)
    store_tool_registry(None, tools)
    assert get_tool_registry("") is None


def test_store_tool_registry_empty_tools():
    store_tool_registry("s1", [])
    store_tool_registry("s1", None)
    assert get_tool_registry("s1") is None


def test_store_tool_registry_non_iterable():
    store_tool_registry("s1", 42)
    assert get_tool_registry("s1") is None


def test_store_tool_registry_with_attr_objects():
    class Fn:
        name = "my_tool"
        description = "does stuff"
        parameters = {"type": "object"}

    class Tool:
        type = "function"
        function = Fn()

    store_tool_registry("s1", [Tool()])
    reg = get_tool_registry("s1")
    assert reg is not None
    assert "my_tool" in reg
    td = reg["my_tool"]
    assert isinstance(td, dict)
    assert td["function"]["name"] == "my_tool"


def test_store_tool_registry_skips_nameless():
    tools = [
        {"type": "function", "function": {}},  # No name
        {"type": "function", "function": {"name": "valid"}},
    ]
    store_tool_registry("s1", tools)
    reg = get_tool_registry("s1")
    assert len(reg) == 1
    assert "valid" in reg


# ---------------------------------------------------------------------------
# AttachedSession
# ---------------------------------------------------------------------------


def test_attached_session_no_context():
    with AttachedSession("nonexistent") as sess:
        assert sess._token is None


def test_attached_session_with_context():
    ctx = otel_context.get_current()
    store_context("s1", ctx)
    with AttachedSession("s1") as sess:
        assert sess._token is not None
    # After exit, token is None
    assert sess._token is None


def test_attached_session_none_sid():
    # With no stored context, should be a no-op
    with AttachedSession(None):
        pass
