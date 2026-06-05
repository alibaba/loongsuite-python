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

"""Tests for MiniSweAgentInstrumentor lifecycle (instrument / uninstrument).

These are integration-style tests that verify the instrumentor can be
imported, instantiated, and run through its instrument/uninstrument
cycle against the stub modules injected by ``conftest.py``.

Because ``mini-swe-agent`` is not actually installed, ``BaseInstrumentor.instrument()``
detects a dependency conflict and returns early.  For lifecycle tests that need the
wrapping to actually happen, we call ``_instrument()`` / ``_uninstrument()`` directly
or patch the dependency check.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# =====================================================================
# Import / metadata
# =====================================================================


class TestImportAndMetadata:
    """Verify basic module-level attributes."""

    def test_import_instrumentor_class(self):
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        assert MiniSweAgentInstrumentor is not None

    def test_version_string(self):
        from packaging.version import Version

        from opentelemetry.instrumentation.minisweagent.version import (
            __version__,
        )

        assert isinstance(__version__, str)
        parsed_version = Version(__version__)
        assert parsed_version.release

    def test_instruments_tuple(self):
        from opentelemetry.instrumentation.minisweagent.package import (
            _instruments,
        )

        assert isinstance(_instruments, tuple)
        assert len(_instruments) >= 1
        assert "mini-swe-agent" in _instruments[0]

    def test_supports_metrics(self):
        from opentelemetry.instrumentation.minisweagent.package import (
            _supports_metrics,
        )

        assert _supports_metrics is True

    def test_instrumentation_dependencies(self):
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        deps = MiniSweAgentInstrumentor().instrumentation_dependencies()
        assert isinstance(deps, tuple)
        assert len(deps) >= 1


# =====================================================================
# Instrument / uninstrument lifecycle
# =====================================================================


class TestInstrumentLifecycle:
    """Verify _instrument() and _uninstrument() don't raise.

    We bypass the ``BaseInstrumentor.instrument()`` dependency check
    by calling ``_instrument()`` directly.
    """

    def test_instrument_does_not_raise(self):
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        inst = MiniSweAgentInstrumentor()
        inst._instrument()
        inst._uninstrument()

    def test_double_instrument(self):
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        inst = MiniSweAgentInstrumentor()
        inst._instrument()
        inst._instrument()  # second call should be safe
        inst._uninstrument()

    def test_uninstrument_without_instrument(self):
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        inst = MiniSweAgentInstrumentor()
        # uninstrument without prior instrument should be a no-op
        inst._uninstrument()

    def test_instrument_with_tracer_provider(self):
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        inst = MiniSweAgentInstrumentor()
        inst._instrument(tracer_provider=provider)
        inst._uninstrument()

    def test_public_instrument_with_dependency_conflict_logs_and_returns(self):
        """``instrument()`` should not raise even when the dependency is missing."""
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        inst = MiniSweAgentInstrumentor()
        # This should log an error and return silently
        inst.instrument()
        inst.uninstrument()


# =====================================================================
# get_environment wrapping
# =====================================================================


class TestGetEnvironmentWrapping:
    """Verify that ``get_environment`` is wrapped/unwrapped correctly."""

    def test_get_environment_wrapped_after_instrument(self):
        import minisweagent.environments as envs_mod

        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        original = envs_mod.get_environment
        inst = MiniSweAgentInstrumentor()
        inst._instrument()

        # After instrumentation, get_environment should be replaced
        assert envs_mod.get_environment is not original

        inst._uninstrument()

    def test_get_environment_restored_after_uninstrument(self):
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        inst = MiniSweAgentInstrumentor()
        inst._instrument()
        inst._uninstrument()

        # After uninstrumentation, the class-level cache should be cleared.
        assert MiniSweAgentInstrumentor._original_get_environment is None

    def test_wrapped_get_environment_returns_tracing_environment(self):
        import minisweagent.environments as envs_mod

        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            TracingEnvironment,
        )

        inst = MiniSweAgentInstrumentor()
        inst._instrument()

        # Call the wrapped get_environment
        env = envs_mod.get_environment({"environment_class": "local"})
        assert isinstance(env, TracingEnvironment)

        inst._uninstrument()


# =====================================================================
# DefaultAgent wrapping
# =====================================================================


class TestDefaultAgentWrapping:
    """Verify DefaultAgent.run / step are wrapped/unwrapped."""

    def test_default_agent_run_wrapped(self):
        from minisweagent.agents.default import DefaultAgent

        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        inst = MiniSweAgentInstrumentor()
        inst._instrument()

        # After instrumentation, run should have __wrapped__
        assert hasattr(DefaultAgent.run, "__wrapped__")

        inst._uninstrument()

    def test_default_agent_step_wrapped(self):
        from minisweagent.agents.default import DefaultAgent

        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        inst = MiniSweAgentInstrumentor()
        inst._instrument()

        assert hasattr(DefaultAgent.step, "__wrapped__")

        inst._uninstrument()

    def test_default_agent_unwrapped(self):
        from minisweagent.agents.default import DefaultAgent

        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        inst = MiniSweAgentInstrumentor()
        inst._instrument()
        inst._uninstrument()

        # After uninstrumentation, __wrapped__ should be removed
        assert not hasattr(DefaultAgent.run, "__wrapped__")
        assert not hasattr(DefaultAgent.step, "__wrapped__")


# =====================================================================
# CLI app patching
# =====================================================================


class TestCliAppPatching:
    """Verify mini CLI app is patched/unpatched."""

    def test_app_patched_after_instrument(self):
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            _PATCH_FLAG,
            _MiniTyperAppProxy,
        )

        inst = MiniSweAgentInstrumentor()
        inst._instrument()

        mini_mod = sys.modules.get("minisweagent.run.mini")
        assert mini_mod is not None
        assert isinstance(mini_mod.app, _MiniTyperAppProxy)
        assert getattr(mini_mod, _PATCH_FLAG, False) is True

        inst._uninstrument()

    def test_app_unpatched_after_uninstrument(self):
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            _PATCH_FLAG,
            _MiniTyperAppProxy,
        )

        inst = MiniSweAgentInstrumentor()
        inst._instrument()
        inst._uninstrument()

        mini_mod = sys.modules.get("minisweagent.run.mini")
        assert mini_mod is not None
        # app should be restored and patch flag removed
        assert not isinstance(mini_mod.app, _MiniTyperAppProxy)
        assert not getattr(mini_mod, _PATCH_FLAG, False)


# =====================================================================
# Agent wrapper helpers (unit tests for _task_preview, etc.)
# =====================================================================


class TestAgentWrapperHelpers:
    """Tests for helper functions in agent_wrappers module."""

    def test_task_preview_short_string(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            _task_preview,
        )

        assert _task_preview("short task") == "short task"

    def test_task_preview_empty_string(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            _task_preview,
        )

        assert _task_preview("") == ""

    def test_task_preview_long_string_truncated(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            _task_preview,
        )

        long_task = "x" * 300
        result = _task_preview(long_task)
        assert result.endswith("...")
        assert len(result) <= 256

    def test_task_preview_exact_limit(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            _task_preview,
        )

        task = "y" * 256
        result = _task_preview(task)
        # Exactly at the limit, should not be truncated
        assert result == task

    def test_task_preview_one_over_limit(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            _task_preview,
        )

        task = "z" * 257
        result = _task_preview(task)
        assert result.endswith("...")
        assert len(result) == 256

    def test_request_model_from_agent(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            _request_model_from_agent,
        )

        class Agent:
            class model:
                class config:
                    model_name = "gpt-4o"

        assert _request_model_from_agent(Agent()) == "gpt-4o"

    def test_request_model_none_model(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            _request_model_from_agent,
        )

        class Agent:
            model = None

        assert _request_model_from_agent(Agent()) is None

    def test_request_model_none_config(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            _request_model_from_agent,
        )

        class Agent:
            class model:
                config = None

        assert _request_model_from_agent(Agent()) is None

    def test_request_model_no_model_name(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            _request_model_from_agent,
        )

        class Agent:
            class model:
                class config:
                    model_name = None

        assert _request_model_from_agent(Agent()) is None


# =====================================================================
# DefaultAgentStepWrapper._limits_exceeded
# =====================================================================


class TestLimitsExceeded:
    """Tests for DefaultAgentStepWrapper._limits_exceeded."""

    def test_no_config(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            DefaultAgentStepWrapper,
        )

        class Agent:
            config = None

        assert DefaultAgentStepWrapper._limits_exceeded(Agent()) is False

    def test_step_limit_not_exceeded(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            DefaultAgentStepWrapper,
        )

        class Agent:
            class config:
                step_limit = 10
                cost_limit = 0

            n_calls = 5
            cost = 0

        assert DefaultAgentStepWrapper._limits_exceeded(Agent()) is False

    def test_step_limit_exceeded(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            DefaultAgentStepWrapper,
        )

        class Agent:
            class config:
                step_limit = 10
                cost_limit = 0

            n_calls = 10
            cost = 0

        assert DefaultAgentStepWrapper._limits_exceeded(Agent()) is True

    def test_step_limit_exceeded_over(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            DefaultAgentStepWrapper,
        )

        class Agent:
            class config:
                step_limit = 5
                cost_limit = 0

            n_calls = 20
            cost = 0

        assert DefaultAgentStepWrapper._limits_exceeded(Agent()) is True

    def test_cost_limit_exceeded(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            DefaultAgentStepWrapper,
        )

        class Agent:
            class config:
                step_limit = 0
                cost_limit = 5.0

            n_calls = 0
            cost = 5.0

        assert DefaultAgentStepWrapper._limits_exceeded(Agent()) is True

    def test_cost_limit_not_exceeded(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            DefaultAgentStepWrapper,
        )

        class Agent:
            class config:
                step_limit = 0
                cost_limit = 5.0

            n_calls = 0
            cost = 3.0

        assert DefaultAgentStepWrapper._limits_exceeded(Agent()) is False

    def test_zero_limits_means_no_limit(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            DefaultAgentStepWrapper,
        )

        class Agent:
            class config:
                step_limit = 0
                cost_limit = 0

            n_calls = 999
            cost = 999.0

        assert DefaultAgentStepWrapper._limits_exceeded(Agent()) is False

    def test_none_limits_treated_as_zero(self):
        from opentelemetry.instrumentation.minisweagent.internal.agent_wrappers import (
            DefaultAgentStepWrapper,
        )

        class Agent:
            class config:
                step_limit = None
                cost_limit = None

            n_calls = 100
            cost = 100.0

        assert DefaultAgentStepWrapper._limits_exceeded(Agent()) is False


# =====================================================================
# TracingEnvironment (delegates.py)
# =====================================================================


class TestTracingEnvironment:
    """Tests for the TracingEnvironment delegate wrapper."""

    def test_getattr_delegates_to_inner(self, stub_environment):
        from opentelemetry import trace as trace_api
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            TracingEnvironment,
        )

        tracer = trace_api.get_tracer("test")
        stub_environment.custom_attr = "hello"
        tracing_env = TracingEnvironment(stub_environment, tracer)
        assert tracing_env.custom_attr == "hello"

    def test_execute_delegates_and_returns_result(self, stub_environment):
        from opentelemetry import trace as trace_api
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            TracingEnvironment,
        )

        tracer = trace_api.get_tracer("test")
        tracing_env = TracingEnvironment(stub_environment, tracer)
        result = tracing_env.execute({"command": "ls"}, cwd="/tmp")
        assert result == {"output": "ok", "exit_code": 0}

    def test_execute_handles_non_dict_result(self):
        from opentelemetry import trace as trace_api
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            TracingEnvironment,
        )

        class NonDictEnv:
            def execute(self, action, cwd="", **kwargs):
                return "plain string"

        tracer = trace_api.get_tracer("test")
        tracing_env = TracingEnvironment(NonDictEnv(), tracer)
        result = tracing_env.execute({"command": "echo"})
        assert result == "plain string"

    def test_execute_propagates_exception(self):
        from opentelemetry import trace as trace_api
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            TracingEnvironment,
        )

        class FailEnv:
            def execute(self, action, cwd="", **kwargs):
                raise RuntimeError("command failed")

        tracer = trace_api.get_tracer("test")
        tracing_env = TracingEnvironment(FailEnv(), tracer)
        with pytest.raises(RuntimeError, match="command failed"):
            tracing_env.execute({"command": "fail"})

    def test_execute_propagates_interrupt_agent_flow(self):
        from minisweagent.exceptions import InterruptAgentFlow

        from opentelemetry import trace as trace_api
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            TracingEnvironment,
        )

        class InterruptEnv:
            def execute(self, action, cwd="", **kwargs):
                raise InterruptAgentFlow("interrupted")

        tracer = trace_api.get_tracer("test")
        tracing_env = TracingEnvironment(InterruptEnv(), tracer)
        with pytest.raises(InterruptAgentFlow):
            tracing_env.execute({"command": "interrupt"})

    def test_execute_with_non_dict_action(self):
        from opentelemetry import trace as trace_api
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            TracingEnvironment,
        )

        class AnyEnv:
            def execute(self, action, cwd="", **kwargs):
                return {"done": True}

        tracer = trace_api.get_tracer("test")
        tracing_env = TracingEnvironment(AnyEnv(), tracer)
        # Non-dict action: command extraction should default to ""
        result = tracing_env.execute("not a dict")
        assert result == {"done": True}


# =====================================================================
# _sanitize_tool_result (delegates.py)
# =====================================================================


class TestSanitizeToolResult:
    """Tests for ``_sanitize_tool_result``."""

    def test_normal_dict(self):
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            _sanitize_tool_result,
        )

        result = _sanitize_tool_result({"key": "value", "num": 42})
        assert result == {"key": "value", "num": 42}

    def test_dict_with_nested_structures(self):
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            _sanitize_tool_result,
        )

        data = {"nested": {"a": 1}, "list": [1, 2, 3]}
        result = _sanitize_tool_result(data)
        assert result == data

    def test_non_serializable_value_uses_default_str(self):
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            _sanitize_tool_result,
        )

        # default=str should handle non-serializable types
        result = _sanitize_tool_result({"key": object()})
        assert "key" in result
        assert isinstance(result["key"], str)

    def test_empty_dict(self):
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            _sanitize_tool_result,
        )

        assert _sanitize_tool_result({}) == {}


# =====================================================================
# _MiniTyperAppProxy (cli_wrappers.py)
# =====================================================================


class TestMiniTyperAppProxy:
    """Tests for the _MiniTyperAppProxy wrapper."""

    def test_proxy_delegates_call(self):
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            _MiniTyperAppProxy,
        )

        inner = MagicMock()
        inner.return_value = "result"
        proxy = _MiniTyperAppProxy(inner)
        result = proxy()
        assert result == "result"

    def test_proxy_getattr_forwards(self):
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            _MiniTyperAppProxy,
        )

        inner = MagicMock()
        inner.some_command = "cmd_value"
        proxy = _MiniTyperAppProxy(inner)
        assert proxy.some_command == "cmd_value"

    def test_proxy_call_propagates_exception(self):
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            _MiniTyperAppProxy,
        )

        inner = MagicMock()
        inner.side_effect = ValueError("boom")
        proxy = _MiniTyperAppProxy(inner)
        with pytest.raises(ValueError, match="boom"):
            proxy()

    def test_proxy_call_propagates_base_exception(self):
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            _MiniTyperAppProxy,
        )

        inner = MagicMock()
        inner.side_effect = KeyboardInterrupt()
        proxy = _MiniTyperAppProxy(inner)
        with pytest.raises(KeyboardInterrupt):
            proxy()


# =====================================================================
# patch / unpatch CLI module functions
# =====================================================================


class TestPatchUnpatchCliModule:
    """Tests for patch_mini_cli_app_module / unpatch_mini_cli_app_module."""

    def test_patch_and_unpatch_roundtrip(self):
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            _PATCH_FLAG,
            _MiniTyperAppProxy,
            patch_mini_cli_app_module,
            unpatch_mini_cli_app_module,
        )

        mini_mod = sys.modules["minisweagent.run.mini"]

        patch_mini_cli_app_module()
        assert isinstance(mini_mod.app, _MiniTyperAppProxy)
        assert getattr(mini_mod, _PATCH_FLAG) is True

        unpatch_mini_cli_app_module()
        assert not isinstance(mini_mod.app, _MiniTyperAppProxy)
        assert not getattr(mini_mod, _PATCH_FLAG, False)

    def test_patch_idempotent(self):
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            patch_mini_cli_app_module,
            unpatch_mini_cli_app_module,
        )

        patch_mini_cli_app_module()
        first_proxy = sys.modules["minisweagent.run.mini"].app
        patch_mini_cli_app_module()  # second call should be no-op
        assert sys.modules["minisweagent.run.mini"].app is first_proxy

        unpatch_mini_cli_app_module()

    def test_unpatch_without_patch_is_noop(self):
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            unpatch_mini_cli_app_module,
        )

        # Should not raise
        unpatch_mini_cli_app_module()

    def test_patch_updates_get_environment_reference(self):
        """After patching, minisweagent.run.mini.get_environment should
        point to minisweagent.environments.get_environment."""
        import minisweagent.environments as envs_mod

        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            patch_mini_cli_app_module,
            unpatch_mini_cli_app_module,
        )

        patch_mini_cli_app_module()
        mini_mod = sys.modules["minisweagent.run.mini"]
        assert mini_mod.get_environment is envs_mod.get_environment

        unpatch_mini_cli_app_module()
