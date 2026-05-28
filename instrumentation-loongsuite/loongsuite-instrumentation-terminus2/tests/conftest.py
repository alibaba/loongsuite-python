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

"""Test configuration for Terminus2 instrumentation tests.

Injects lightweight stub modules for ``terminal_bench.agents.terminus_2.*``
into ``sys.modules`` so that ``wrapt.wrap_function_wrapper`` can resolve
patch targets without installing terminal-bench.

Stub methods delegate to instance-level ``_*_override`` / ``_*_error``
attributes so that tests can control return values and trigger exceptions
*after* the instrumentation wrapper has captured the original method.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

# ---------------------------------------------------------------------------
# Ensure workspace packages are on sys.path
# ---------------------------------------------------------------------------

_TERMINUS2_SRC = Path(__file__).resolve().parents[1] / "src"
if _TERMINUS2_SRC.is_dir() and str(_TERMINUS2_SRC) not in sys.path:
    sys.path.insert(0, str(_TERMINUS2_SRC))

# ---------------------------------------------------------------------------
# Stub data classes
# ---------------------------------------------------------------------------


@dataclass
class Command:
    """Stub for ``terminal_bench`` Command object."""

    keystrokes: str = ""
    duration_sec: Optional[float] = None


@dataclass
class ParseResult:
    """Stub for the result returned by parser.parse_response."""

    commands: list = field(default_factory=list)
    is_task_complete: bool = False
    error: Optional[str] = None
    warning: Optional[str] = None


@dataclass
class AgentResult:
    """Stub for the result returned by Terminus2.perform_task."""

    failure_mode: Any = None
    timestamped_markers: list = field(default_factory=list)


class Chat:
    """Stub for the Chat object passed to _run_agent_loop."""

    def __init__(self, messages: Optional[list] = None):
        self._messages = messages or []


# ---------------------------------------------------------------------------
# Stub classes for terminal-bench modules
#
# Method behaviour is controlled per-instance via:
#   _<method>_override  -- if set, returned instead of the default
#   _<method>_error     -- if set (an Exception), raised before returning
# This allows tests to configure behaviour after instrumentation wrapping.
# ---------------------------------------------------------------------------


class Terminus2:
    """Stub Terminus2 agent class."""

    _model_name: str = "gpt-4o"
    _parser_name: str = "json"
    _prompt_template: str = "You are a helpful terminal agent."
    _pending_completion: bool = False

    # -- per-instance overrides (set by tests) --
    _perform_task_override: Any = None
    _perform_task_error: Optional[Exception] = None

    _run_agent_loop_override: Any = None
    _run_agent_loop_error: Optional[Exception] = None

    _execute_commands_override: Any = None
    _execute_commands_error: Optional[Exception] = None

    _handle_llm_override: Any = None
    _handle_llm_error: Optional[Exception] = None

    _summarize_error: Optional[Exception] = None

    def perform_task(self, instruction: str) -> AgentResult:
        if self._perform_task_error is not None:
            raise self._perform_task_error
        if self._perform_task_override is not None:
            return self._perform_task_override
        return AgentResult()

    def _run_agent_loop(
        self,
        initial_prompt: str,
        session: Any = None,
        chat: Any = None,
        logging_dir: Optional[str] = None,
        original_instruction: str = "",
    ) -> None:
        if self._run_agent_loop_error is not None:
            raise self._run_agent_loop_error
        if self._run_agent_loop_override is not None:
            return self._run_agent_loop_override
        return None

    def _execute_commands(self, commands: list) -> tuple:
        if self._execute_commands_error is not None:
            raise self._execute_commands_error
        if self._execute_commands_override is not None:
            return self._execute_commands_override
        return (False, "command output")

    def _handle_llm_interaction(self, *args: Any, **kwargs: Any) -> tuple:
        if self._handle_llm_error is not None:
            raise self._handle_llm_error
        if self._handle_llm_override is not None:
            return self._handle_llm_override
        return ([], False, "")

    def _summarize(self, *args: Any, **kwargs: Any) -> Any:
        if self._summarize_error is not None:
            raise self._summarize_error
        return None


class TerminusJSONPlainParser:
    """Stub JSON parser class."""

    _parse_response_override: Any = None
    _parse_response_error: Optional[Exception] = None

    def parse_response(self, response: str) -> ParseResult:
        if self._parse_response_error is not None:
            raise self._parse_response_error
        if self._parse_response_override is not None:
            return self._parse_response_override
        return ParseResult()


class TerminusXMLPlainParser:
    """Stub XML parser class."""

    _parse_response_override: Any = None
    _parse_response_error: Optional[Exception] = None

    def parse_response(self, response: str) -> ParseResult:
        if self._parse_response_error is not None:
            raise self._parse_response_error
        if self._parse_response_override is not None:
            return self._parse_response_override
        return ParseResult()


# ---------------------------------------------------------------------------
# Inject stub modules into sys.modules
# ---------------------------------------------------------------------------


def _inject_stub_modules():
    """Register fake ``terminal_bench.agents.terminus_2.*`` modules."""

    terminal_bench_mod = types.ModuleType("terminal_bench")
    terminal_bench_agents_mod = types.ModuleType("terminal_bench.agents")
    terminus_2_pkg_mod = types.ModuleType("terminal_bench.agents.terminus_2")
    terminus_2_mod = types.ModuleType(
        "terminal_bench.agents.terminus_2.terminus_2"
    )
    json_parser_mod = types.ModuleType(
        "terminal_bench.agents.terminus_2.terminus_json_plain_parser"
    )
    xml_parser_mod = types.ModuleType(
        "terminal_bench.agents.terminus_2.terminus_xml_plain_parser"
    )

    # Populate terminus_2 module
    terminus_2_mod.Terminus2 = Terminus2

    # Populate parser modules
    json_parser_mod.TerminusJSONPlainParser = TerminusJSONPlainParser
    xml_parser_mod.TerminusXMLPlainParser = TerminusXMLPlainParser

    # Wire up parent references
    terminal_bench_mod.agents = terminal_bench_agents_mod
    terminal_bench_agents_mod.terminus_2 = terminus_2_pkg_mod
    terminus_2_pkg_mod.terminus_2 = terminus_2_mod
    terminus_2_pkg_mod.terminus_json_plain_parser = json_parser_mod
    terminus_2_pkg_mod.terminus_xml_plain_parser = xml_parser_mod

    # Register in sys.modules
    sys.modules["terminal_bench"] = terminal_bench_mod
    sys.modules["terminal_bench.agents"] = terminal_bench_agents_mod
    sys.modules["terminal_bench.agents.terminus_2"] = terminus_2_pkg_mod
    sys.modules["terminal_bench.agents.terminus_2.terminus_2"] = terminus_2_mod
    sys.modules[
        "terminal_bench.agents.terminus_2.terminus_json_plain_parser"
    ] = json_parser_mod
    sys.modules[
        "terminal_bench.agents.terminus_2.terminus_xml_plain_parser"
    ] = xml_parser_mod


# Inject stubs before any test imports the instrumentation module.
_inject_stub_modules()


# ---------------------------------------------------------------------------
# OTel test fixtures
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config):
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"


# Drop cached instrumentation module so it re-imports against stubs.
for _m in list(sys.modules):
    if _m.startswith("opentelemetry.instrumentation.terminus2"):
        del sys.modules[_m]

from opentelemetry.instrumentation.terminus2 import Terminus2Instrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function")
def instrument(tracer_provider):
    """Instrument terminus-2, yield the instrumentor, then uninstrument."""
    instrumentor = Terminus2Instrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()
