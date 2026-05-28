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

"""Lifecycle and smoke tests for ``ClawEvalInstrumentor``."""

from __future__ import annotations

import importlib

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


class TestImport:
    """Verify the instrumentation package is importable."""

    def test_import_instrumentor(self):
        mod = importlib.import_module(
            "opentelemetry.instrumentation.claw_eval"
        )
        assert hasattr(mod, "ClawEvalInstrumentor")

    def test_import_config(self):
        mod = importlib.import_module(
            "opentelemetry.instrumentation.claw_eval.config"
        )
        assert hasattr(mod, "OTEL_INSTRUMENTATION_CLAW_EVAL_ENABLED")
        assert hasattr(mod, "OTEL_CLAW_EVAL_CAPTURE_CONTENT")
        assert hasattr(mod, "OTEL_CLAW_EVAL_PROPAGATE_TO_WORKER")

    def test_import_package(self):
        mod = importlib.import_module(
            "opentelemetry.instrumentation.claw_eval.package"
        )
        assert hasattr(mod, "_instruments")

    def test_import_version(self):
        mod = importlib.import_module(
            "opentelemetry.instrumentation.claw_eval.version"
        )
        assert hasattr(mod, "__version__")


# ---------------------------------------------------------------------------
# Instrumentation dependency declaration
# ---------------------------------------------------------------------------


class TestInstrumentationDependencies:
    """Verify the declared instrumentation dependencies."""

    def test_instrumentation_dependencies_returns_expected(self):
        from opentelemetry.instrumentation.claw_eval import (
            ClawEvalInstrumentor,
        )

        instr = ClawEvalInstrumentor()
        deps = instr.instrumentation_dependencies()
        assert tuple(deps) == ("claw-eval >= 0.1.0",)

    def test_dependencies_match_package_instruments(self):
        from opentelemetry.instrumentation.claw_eval import (
            ClawEvalInstrumentor,
        )
        from opentelemetry.instrumentation.claw_eval.package import (
            _instruments,
        )

        instr = ClawEvalInstrumentor()
        assert tuple(instr.instrumentation_dependencies()) == _instruments


# ---------------------------------------------------------------------------
# Instrument / uninstrument lifecycle
# ---------------------------------------------------------------------------


class TestInstrumentLifecycle:
    """Verify instrument() and uninstrument() can be called without error."""

    def test_instrument_uninstrument(self, tracer_provider):
        from opentelemetry.instrumentation.claw_eval import (
            ClawEvalInstrumentor,
        )

        instr = ClawEvalInstrumentor()
        # Should not raise even though claw_eval modules are mocks.
        instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
        instr.uninstrument()

    def test_double_uninstrument_does_not_raise(self, tracer_provider):
        from opentelemetry.instrumentation.claw_eval import (
            ClawEvalInstrumentor,
        )

        instr = ClawEvalInstrumentor()
        instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
        instr.uninstrument()
        # Second uninstrument should be a no-op.
        instr.uninstrument()

    def test_instrument_wraps_cli_functions(self, tracer_provider):
        """After instrument(), CLI functions should have __wrapped__."""
        import claw_eval.cli as cli

        from opentelemetry.instrumentation.claw_eval import (
            ClawEvalInstrumentor,
        )

        original_cmd_run = cli.cmd_run
        original_cmd_batch = cli.cmd_batch
        original_run_single = cli._run_single_task

        instr = ClawEvalInstrumentor()
        instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            assert hasattr(cli.cmd_run, "__wrapped__")
            assert hasattr(cli.cmd_batch, "__wrapped__")
            assert hasattr(cli._run_single_task, "__wrapped__")
        finally:
            instr.uninstrument()

        # After uninstrument, originals should be restored.
        assert cli.cmd_run is original_cmd_run
        assert cli.cmd_batch is original_cmd_batch
        assert cli._run_single_task is original_run_single

    def test_instrument_wraps_run_task(self, tracer_provider):
        import claw_eval.runner.loop as loop

        from opentelemetry.instrumentation.claw_eval import (
            ClawEvalInstrumentor,
        )

        original = loop.run_task
        instr = ClawEvalInstrumentor()
        instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            assert hasattr(loop.run_task, "__wrapped__")
        finally:
            instr.uninstrument()

        assert loop.run_task is original

    def test_instrument_wraps_provider_chat(self, tracer_provider):
        import claw_eval.runner.providers.openai_compat as oc

        from opentelemetry.instrumentation.claw_eval import (
            ClawEvalInstrumentor,
        )

        instr = ClawEvalInstrumentor()
        instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            assert hasattr(oc.OpenAICompatProvider.chat, "__wrapped__")
        finally:
            instr.uninstrument()

    def test_instrument_wraps_compact(self, tracer_provider):
        import claw_eval.runner.compact as compact

        from opentelemetry.instrumentation.claw_eval import (
            ClawEvalInstrumentor,
        )

        original = compact.do_auto_compact
        instr = ClawEvalInstrumentor()
        instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            assert hasattr(compact.do_auto_compact, "__wrapped__")
        finally:
            instr.uninstrument()

        assert compact.do_auto_compact is original

    def test_instrument_wraps_dispatchers(self, tracer_provider):
        import claw_eval.runner.dispatcher as disp
        import claw_eval.runner.sandbox_dispatcher as sdisp

        from opentelemetry.instrumentation.claw_eval import (
            ClawEvalInstrumentor,
        )

        instr = ClawEvalInstrumentor()
        instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            assert hasattr(disp.ToolDispatcher.dispatch, "__wrapped__")
            assert hasattr(sdisp.SandboxToolDispatcher.dispatch, "__wrapped__")
        finally:
            instr.uninstrument()

    def test_instrument_wraps_judge(self, tracer_provider):
        import claw_eval.graders.llm_judge as lj

        from opentelemetry.instrumentation.claw_eval import (
            ClawEvalInstrumentor,
        )

        instr = ClawEvalInstrumentor()
        instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            assert hasattr(lj.LLMJudge.evaluate, "__wrapped__")
            assert hasattr(lj.LLMJudge.evaluate_actions, "__wrapped__")
            assert hasattr(lj.LLMJudge.evaluate_visual, "__wrapped__")
        finally:
            instr.uninstrument()

    def test_instrument_wraps_grader_loaders(self, tracer_provider):
        import claw_eval.graders.base as base
        import claw_eval.graders.registry as reg

        from opentelemetry.instrumentation.claw_eval import (
            ClawEvalInstrumentor,
        )

        instr = ClawEvalInstrumentor()
        instr.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        try:
            assert hasattr(reg.get_grader, "__wrapped__")
            assert hasattr(base.load_peer_grader, "__wrapped__")
        finally:
            instr.uninstrument()


# ---------------------------------------------------------------------------
# _unwrap helpers
# ---------------------------------------------------------------------------


class TestUnwrapHelpers:
    """Test the module-level _unwrap_func / _unwrap_method helpers."""

    def test_unwrap_func_missing_module(self):
        """_unwrap_func should not raise for a non-existent module."""
        from opentelemetry.instrumentation.claw_eval import _unwrap_func

        _unwrap_func("nonexistent.module.path", "some_func")

    def test_unwrap_func_missing_attr(self):
        """_unwrap_func should not raise for a missing attribute."""
        from opentelemetry.instrumentation.claw_eval import _unwrap_func

        _unwrap_func("claw_eval.cli", "nonexistent_function")

    def test_unwrap_method_missing_class(self):
        """_unwrap_method should not raise for a missing class."""
        from opentelemetry.instrumentation.claw_eval import _unwrap_method

        _unwrap_method("claw_eval.cli", "NonExistentClass", "method")

    def test_unwrap_method_missing_method(self):
        """_unwrap_method should not raise for a missing method."""
        from opentelemetry.instrumentation.claw_eval import _unwrap_method

        _unwrap_method(
            "claw_eval.runner.providers.openai_compat",
            "OpenAICompatProvider",
            "nonexistent_method",
        )
