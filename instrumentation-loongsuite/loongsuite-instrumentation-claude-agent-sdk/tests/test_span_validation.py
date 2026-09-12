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

"""Specific validation tests for Claude Agent SDK instrumentation.

These tests provide detailed validation for specific aspects of the instrumentation:
- Agent span attributes and structure
- LLM span input/output messages
- Tool span attributes and results
- Span hierarchy and timeline
"""

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import MagicMock

import pytest
import yaml

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

# ============================================================================
# Helper Functions
# ============================================================================


def load_cassette(filename: str) -> Dict[str, Any]:
    """Load a test case from cassettes directory."""
    cassette_path = Path(__file__).parent / "cassettes" / filename
    with open(cassette_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_mock_message_from_data(message_data: Dict[str, Any]) -> Any:
    """Create a mock message object from test data dictionary."""
    mock_msg = MagicMock()
    msg_type = message_data["type"]

    mock_msg.__class__.__name__ = msg_type

    if msg_type == "SystemMessage":
        mock_msg.subtype = message_data["subtype"]
        mock_msg.data = message_data["data"]

    elif msg_type == "AssistantMessage":
        mock_msg.model = message_data["model"]
        mock_msg.content = []

        for block_data in message_data["content"]:
            mock_block = MagicMock()
            block_type = block_data["type"]
            mock_block.__class__.__name__ = block_type

            if block_type == "TextBlock":
                mock_block.text = block_data["text"]
            elif block_type == "ToolUseBlock":
                mock_block.id = block_data["id"]
                mock_block.name = block_data["name"]
                mock_block.input = block_data["input"]

            mock_msg.content.append(mock_block)

        mock_msg.parent_tool_use_id = message_data.get("parent_tool_use_id")
        mock_msg.error = message_data.get("error")

    elif msg_type == "UserMessage":
        mock_msg.content = []

        for block_data in message_data["content"]:
            mock_block = MagicMock()
            mock_block.__class__.__name__ = block_data["type"]

            if block_data["type"] == "ToolResultBlock":
                mock_block.tool_use_id = block_data["tool_use_id"]
                mock_block.content = block_data["content"]
                mock_block.is_error = block_data["is_error"]

            mock_msg.content.append(mock_block)

        mock_msg.uuid = message_data.get("uuid")
        mock_msg.parent_tool_use_id = message_data.get("parent_tool_use_id")
        mock_msg.tool_use_result = message_data.get("tool_use_result")

    elif msg_type == "ResultMessage":
        mock_msg.subtype = message_data["subtype"]
        mock_msg.duration_ms = message_data["duration_ms"]
        mock_msg.duration_api_ms = message_data.get("duration_api_ms")
        mock_msg.is_error = message_data["is_error"]
        mock_msg.num_turns = message_data["num_turns"]
        mock_msg.session_id = message_data.get("session_id")
        mock_msg.total_cost_usd = message_data["total_cost_usd"]
        mock_msg.usage = message_data["usage"]
        mock_msg.result = message_data["result"]
        mock_msg.structured_output = message_data.get("structured_output")

    return mock_msg


async def create_mock_stream_from_messages(
    messages: List[Dict[str, Any]],
) -> AsyncIterator[Any]:
    """Create a mock async stream of messages."""
    for message_data in messages:
        yield create_mock_message_from_data(message_data)


def find_agent_span(spans):
    """Find the Agent span."""
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )

    for span in spans:
        attrs = dict(span.attributes or {})
        if attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "invoke_agent":
            return span
    return None


def find_llm_spans(spans):
    """Find all LLM spans."""
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )

    return [
        s
        for s in spans
        if dict(s.attributes or {}).get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "chat"
    ]


def find_tool_spans(spans):
    """Find all Tool spans."""
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )

    return [
        s
        for s in spans
        if dict(s.attributes or {}).get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "execute_tool"
    ]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tracer_provider():
    """Create a tracer provider for testing."""
    return TracerProvider()


@pytest.fixture
def span_exporter(tracer_provider):
    """Create an in-memory span exporter."""
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture
def instrument(tracer_provider):
    """Instrument the Claude Agent SDK."""
    from opentelemetry.instrumentation.claude_agent_sdk import (  # noqa: PLC0415
        ClaudeAgentSDKInstrumentor,
    )

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    yield instrumentor
    instrumentor.uninstrument()


# ============================================================================
# Tests - Agent Span
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cassette_file",
    [
        "test_foo_sh_command.yaml",
        "test_echo_command.yaml",
        "test_pretooluse_hook.yaml",
    ],
)
async def test_agent_span_correctness(
    cassette_file, instrument, span_exporter, tracer_provider
):
    """Verify Agent span correctness.

    Validates:
    1. Agent span exists and is unique
    2. Agent span is a root span (no parent)
    3. Agent span contains correct attributes (operation.name, agent.name, etc.)
    4. Agent span includes token usage statistics
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    test_case = load_cassette(cassette_file)
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    mock_stream = create_mock_stream_from_messages(test_case["messages"])

    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass

    spans = span_exporter.get_finished_spans()
    agent_span = find_agent_span(spans)

    # Verify Agent span exists and is unique
    agent_spans = [
        s
        for s in spans
        if dict(s.attributes or {}).get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    ]
    assert len(agent_spans) == 1, (
        f"Should have exactly one Agent span, got: {len(agent_spans)}"
    )

    # Verify it's a root span
    assert agent_span.parent is None, (
        "Agent span should be a root span with no parent"
    )

    # Verify required attributes
    attrs = dict(agent_span.attributes or {})
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "invoke_agent"

    # Verify token usage statistics
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in attrs, (
        "Should have input_tokens"
    )
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in attrs, (
        "Should have output_tokens"
    )


# ============================================================================
# Tests - LLM Span
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cassette_file",
    [
        "test_foo_sh_command.yaml",
        "test_echo_command.yaml",
        "test_pretooluse_hook.yaml",
    ],
)
async def test_llm_span_correctness(
    cassette_file, instrument, span_exporter, tracer_provider
):
    """Verify LLM span correctness.

    Validates:
    1. LLM spans exist with correct count
    2. LLM spans are children of Agent span
    3. LLM span attributes are correct (model, provider, operation, etc.)
    4. LLM span output.messages have unique tool_call.id (no duplicates)
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    test_case = load_cassette(cassette_file)
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    mock_stream = create_mock_stream_from_messages(test_case["messages"])

    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass

    spans = span_exporter.get_finished_spans()
    agent_span = find_agent_span(spans)
    llm_spans = find_llm_spans(spans)

    # Verify LLM spans exist
    assert len(llm_spans) > 0, "Should have at least one LLM span"

    # Verify all LLM spans are children of Agent span
    for llm_span in llm_spans:
        assert llm_span.parent is not None, "LLM span should have a parent"
        assert llm_span.parent.span_id == agent_span.context.span_id, (
            "LLM span's parent should be Agent span"
        )

        # Verify basic attributes
        attrs = dict(llm_span.attributes or {})
        assert attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
        assert GenAIAttributes.GEN_AI_REQUEST_MODEL in attrs

        # Verify uniqueness of tool_call.id in output.messages
        if GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in attrs:
            output_messages_raw = attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
            if isinstance(output_messages_raw, str):
                output_messages = json.loads(output_messages_raw)
            else:
                output_messages = output_messages_raw

            if isinstance(output_messages, list):
                tool_call_ids = []
                for msg in output_messages:
                    if (
                        isinstance(msg, dict)
                        and msg.get("role") == "assistant"
                    ):
                        parts = msg.get("parts", [])
                        for part in parts:
                            if (
                                isinstance(part, dict)
                                and part.get("type") == "tool_call"
                            ):
                                tool_call_id = part.get("id")
                                if tool_call_id:
                                    assert tool_call_id not in tool_call_ids, (
                                        f"Found duplicate tool_call ID: {tool_call_id}"
                                    )
                                    tool_call_ids.append(tool_call_id)


# ============================================================================
# Tests - Tool Span
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cassette_file",
    [
        "test_foo_sh_command.yaml",
        "test_echo_command.yaml",
        "test_pretooluse_hook.yaml",
    ],
)
async def test_tool_span_correctness(
    cassette_file, instrument, span_exporter, tracer_provider
):
    """Verify Tool span correctness.

    Validates:
    1. Tool spans exist with correct count
    2. Tool spans are children of Agent span (not LLM span)
    3. Tool span attributes are correct (tool.name, tool.call.id, arguments, result, etc.)
    4. Tool span contains correct is_error status
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    test_case = load_cassette(cassette_file)
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    mock_stream = create_mock_stream_from_messages(test_case["messages"])

    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass

    spans = span_exporter.get_finished_spans()
    agent_span = find_agent_span(spans)
    llm_spans = find_llm_spans(spans)
    tool_spans = find_tool_spans(spans)

    # Verify Tool spans exist
    assert len(tool_spans) > 0, "Should have at least one Tool span"

    # Verify all Tool spans are children of Agent span (not LLM span)
    for tool_span in tool_spans:
        assert tool_span.parent is not None, "Tool span should have a parent"
        assert tool_span.parent.span_id == agent_span.context.span_id, (
            "Tool span's parent should be Agent span, not LLM span"
        )

        # Ensure it's not a child of LLM span
        for llm_span in llm_spans:
            assert tool_span.parent.span_id != llm_span.context.span_id, (
                "Tool span should not be a child of LLM span"
            )

        # Verify basic attributes
        attrs = dict(tool_span.attributes or {})
        assert (
            attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "execute_tool"
        )
        assert GenAIAttributes.GEN_AI_TOOL_NAME in attrs, (
            "Should have tool.name"
        )
        assert GenAIAttributes.GEN_AI_TOOL_CALL_ID in attrs, (
            "Should have tool.call.id"
        )


# ============================================================================
# Tests - Span Hierarchy
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cassette_file",
    [
        "test_foo_sh_command.yaml",
        "test_echo_command.yaml",
        "test_pretooluse_hook.yaml",
    ],
)
async def test_span_hierarchy_correctness(
    cassette_file, instrument, span_exporter, tracer_provider
):
    """Verify span hierarchy correctness.

    Validates:
    1. Agent span is the root span
    2. LLM spans are children of Agent span
    3. Tool spans are children of Agent span (not LLM span)
    4. Span timeline is sequential (LLM → Tool → LLM)
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    test_case = load_cassette(cassette_file)
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    mock_stream = create_mock_stream_from_messages(test_case["messages"])

    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass

    spans = span_exporter.get_finished_spans()
    agent_span = find_agent_span(spans)
    llm_spans = find_llm_spans(spans)
    tool_spans = find_tool_spans(spans)

    # Verify Agent span is root
    assert agent_span is not None, "Should have Agent span"
    assert agent_span.parent is None, "Agent span should be root span"

    # Verify LLM spans are children of Agent span
    assert len(llm_spans) > 0, "Should have at least one LLM span"
    for llm_span in llm_spans:
        assert llm_span.parent is not None, "LLM span should have a parent"
        assert llm_span.parent.span_id == agent_span.context.span_id, (
            "LLM span's parent should be Agent span"
        )

    # Verify Tool spans are children of Agent span
    assert len(tool_spans) > 0, "Should have at least one Tool span"
    for tool_span in tool_spans:
        assert tool_span.parent is not None, "Tool span should have a parent"
        assert tool_span.parent.span_id == agent_span.context.span_id, (
            "Tool span's parent should be Agent span"
        )

        # Ensure it's not a child of LLM span
        for llm_span in llm_spans:
            assert tool_span.parent.span_id != llm_span.context.span_id, (
                "Tool span should not be a child of LLM span"
            )


# ============================================================================
# Tests - Skill Tool Span (gen_ai.skill.* attributes)
# ============================================================================


def _write_probe_skill_md(
    project_dir: Path,
    skill_name: str = "probe-skill",
    version: str = "1.2.3",
) -> str:
    """Create a project-level probe SKILL.md and return its project dir."""
    skill_dir = project_dir / ".claude" / "skills" / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\n"
        f"name: {skill_name}\n"
        f"description: Skill telemetry probe for {skill_name}.\n"
        f"version: {version}\n"
        "---\n\n"
        "When this skill is loaded, answer exactly: PROBE_SKILL_MARKER\n",
        encoding="utf-8",
    )
    return str(project_dir)


def _skill_load_messages(
    cwd: str,
    skill_name: str = "probe-skill",
    session_id: str = "skill-session-0001",
    tool_use_id: str = "call_skill_load_probe",
    marker: str = "PROBE_SKILL_MARKER",
) -> List[Dict[str, Any]]:
    """Message sequence for a Skill load, modelled on the SDK message stream."""
    return [
        {
            "type": "SystemMessage",
            "subtype": "init",
            "data": {
                "type": "system",
                "subtype": "init",
                "cwd": cwd,
                "session_id": session_id,
                "tools": ["Skill", "Bash", "Read"],
                "skills": [skill_name],
                "model": "qwen-plus",
                "permissionMode": "bypassPermissions",
                "apiKeySource": "ANTHROPIC_API_KEY",
                "claude_code_version": "2.1.1",
                "output_style": "default",
                "agents": [],
                "slash_commands": [],
                "plugins": [],
                "mcp_servers": [],
                "uuid": "skill-init-uuid",
            },
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "ToolUseBlock",
                    "id": tool_use_id,
                    "name": "Skill",
                    "input": {"skill": skill_name},
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "UserMessage",
            "content": [
                {
                    "type": "ToolResultBlock",
                    "tool_use_id": tool_use_id,
                    "content": f"Launching skill: {skill_name}",
                    "is_error": False,
                }
            ],
            "uuid": "skill-result-uuid",
            "parent_tool_use_id": None,
            "tool_use_result": {
                "success": True,
                "commandName": skill_name,
            },
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [{"type": "TextBlock", "text": marker}],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "ResultMessage",
            "subtype": "success",
            "duration_ms": 3210,
            "duration_api_ms": 9000,
            "is_error": False,
            "num_turns": 2,
            "session_id": session_id,
            "total_cost_usd": 0.012,
            "usage": {
                "input_tokens": 1024,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 32,
                "server_tool_use": {
                    "web_search_requests": 0,
                    "web_fetch_requests": 0,
                },
                "service_tier": "standard",
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 0,
                    "ephemeral_5m_input_tokens": 0,
                },
            },
            "result": marker,
            "structured_output": None,
        },
    ]


@pytest.mark.asyncio
async def test_skill_tool_span_attributes(
    instrument, span_exporter, tracer_provider, tmp_path
):
    """Verify gen_ai.skill.* attributes on a Skill load execute_tool span.

    Validates per the Skill telemetry spec:
    1. Exactly one gen_ai.tool.name=Skill execute_tool span exists.
    2. That span carries gen_ai.skill.name/id/description/version.
    3. skill id is ``claude:project:<skill-name>``.
    4. Metadata is read best-effort from the project SKILL.md frontmatter.
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    cwd = _write_probe_skill_md(tmp_path)
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    mock_stream = create_mock_stream_from_messages(_skill_load_messages(cwd))

    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=(
            "Use the probe-skill Skill tool first. Then answer exactly "
            "PROBE_SKILL_MARKER and nothing else."
        ),
    ):
        pass

    spans = span_exporter.get_finished_spans()

    skill_tool_spans = [
        s
        for s in spans
        if dict(s.attributes or {}).get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "execute_tool"
        and dict(s.attributes or {}).get(GenAIAttributes.GEN_AI_TOOL_NAME)
        == "Skill"
    ]

    # Pass criterion 2: exactly one gen_ai.tool.name=Skill execute_tool span.
    assert len(skill_tool_spans) == 1, (
        f"Should capture exactly one Skill execute_tool span, got "
        f"{len(skill_tool_spans)}"
    )

    tool_span = skill_tool_spans[0]
    attrs = dict(tool_span.attributes or {})

    # Pass criterion 3: span carries all four gen_ai.skill.* attributes.
    assert attrs.get("gen_ai.skill.name") == "probe-skill"
    assert attrs.get("gen_ai.skill.id") == "claude:project:probe-skill"
    assert attrs.get("gen_ai.skill.description") == (
        "Skill telemetry probe for probe-skill."
    )
    assert attrs.get("gen_ai.skill.version") == "1.2.3"

    # Tool span still carries the standard tool attributes.
    assert attrs.get(GenAIAttributes.GEN_AI_TOOL_CALL_ID) == (
        "call_skill_load_probe"
    )


@pytest.mark.asyncio
async def test_skill_metadata_read_failure_does_not_break_sdk(
    instrument, span_exporter, tracer_provider, tmp_path
):
    """Skill metadata read failures must not affect the SDK call (best-effort).

    When cwd points nowhere useful (no SKILL.md), the Skill tool span is still
    created with skill.name/id derived from the tool input; no exception escapes.
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    # cwd with no .claude/skills tree -> SKILL.md read returns empty best-effort
    cwd = str(tmp_path)
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    mock_stream = create_mock_stream_from_messages(_skill_load_messages(cwd))

    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt="Use the probe-skill Skill tool.",
    ):
        pass

    spans = span_exporter.get_finished_spans()
    skill_tool_spans = [
        s
        for s in spans
        if dict(s.attributes or {}).get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "execute_tool"
        and dict(s.attributes or {}).get(GenAIAttributes.GEN_AI_TOOL_NAME)
        == "Skill"
    ]
    assert len(skill_tool_spans) == 1
    attrs = dict(skill_tool_spans[0].attributes or {})
    # name/id fall back to the requested skill; description/version absent.
    assert attrs.get("gen_ai.skill.name") == "probe-skill"
    assert attrs.get("gen_ai.skill.id") == "claude:project:probe-skill"
    assert "gen_ai.skill.description" not in attrs
    assert "gen_ai.skill.version" not in attrs


@pytest.mark.asyncio
async def test_parallel_skill_loads_keep_metadata_isolated(
    instrument, span_exporter, tracer_provider, tmp_path
):
    """Parallel streams with the same tool_use_id must not mix Skill spans."""
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )
    from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (  # noqa: PLC0415
        GEN_AI_SESSION_ID,
    )

    async def interleaved_stream(messages):
        for message in messages:
            await asyncio.sleep(0)
            yield create_mock_message_from_data(message)

    async def run_case(skill_name: str, session_id: str, version: str) -> None:
        project_dir = tmp_path / skill_name
        cwd = _write_probe_skill_md(
            project_dir, skill_name=skill_name, version=version
        )
        messages = _skill_load_messages(
            cwd,
            skill_name=skill_name,
            session_id=session_id,
            tool_use_id="shared_skill_tool",
            marker=f"{skill_name.upper()}_MARKER",
        )
        handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)

        async for _ in _process_agent_invocation_stream(
            wrapped_stream=interleaved_stream(messages),
            handler=handler,
            model="qwen-plus",
            prompt=f"Use the {skill_name} Skill tool.",
        ):
            pass

    await asyncio.gather(
        run_case("alpha-skill", "session-alpha", "1.0.0"),
        run_case("beta-skill", "session-beta", "2.0.0"),
    )

    skill_tool_attrs = []
    for span in span_exporter.get_finished_spans():
        attrs = dict(span.attributes or {})
        if (
            attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "execute_tool"
            and attrs.get(GenAIAttributes.GEN_AI_TOOL_NAME) == "Skill"
        ):
            skill_tool_attrs.append(attrs)

    assert len(skill_tool_attrs) == 2
    attrs_by_skill = {
        attrs["gen_ai.skill.name"]: attrs for attrs in skill_tool_attrs
    }
    assert set(attrs_by_skill) == {"alpha-skill", "beta-skill"}

    assert attrs_by_skill["alpha-skill"]["gen_ai.skill.id"] == (
        "claude:project:alpha-skill"
    )
    assert attrs_by_skill["alpha-skill"]["gen_ai.skill.version"] == "1.0.0"
    assert attrs_by_skill["alpha-skill"][GEN_AI_SESSION_ID] == (
        "session-alpha"
    )

    assert attrs_by_skill["beta-skill"]["gen_ai.skill.id"] == (
        "claude:project:beta-skill"
    )
    assert attrs_by_skill["beta-skill"]["gen_ai.skill.version"] == "2.0.0"
    assert attrs_by_skill["beta-skill"][GEN_AI_SESSION_ID] == "session-beta"
