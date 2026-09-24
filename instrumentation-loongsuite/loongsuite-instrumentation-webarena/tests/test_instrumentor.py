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

"""Instrumentor lifecycle tests for WebarenaInstrumentor.

Covers instrument / uninstrument, dependency declarations, double
instrument / uninstrument, and wrapping verification.
"""

from __future__ import annotations

import sys

from opentelemetry.instrumentation.webarena import WebarenaInstrumentor
from opentelemetry.instrumentation.webarena.package import _instruments
from opentelemetry.instrumentation.webarena.version import __version__


class TestInstrumentorLifecycle:
    """Verify instrument / uninstrument does not crash and is idempotent."""

    def test_instrument_and_uninstrument(self, tracer_provider):
        instrumentor = WebarenaInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )
        instrumentor.uninstrument()

    def test_double_uninstrument_is_safe(self, tracer_provider):
        instrumentor = WebarenaInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )
        instrumentor.uninstrument()
        # Second uninstrument should not raise
        instrumentor.uninstrument()

    def test_instrumentation_dependencies(self):
        instrumentor = WebarenaInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert len(deps) >= 1
        assert any("webarena" in d for d in deps)

    def test_dependencies_match_package(self):
        instrumentor = WebarenaInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert deps == _instruments

    def test_version_is_string(self):
        assert isinstance(__version__, str)
        assert len(__version__) > 0


class TestWrapping:
    """Verify that instrumentation actually wraps the target functions."""

    def test_env_reset_is_wrapped(self, instrument):
        import browser_env.envs as envs_mod

        envs_mod.ScriptBrowserEnv()
        # After instrumentation, the method should have __wrapped__
        assert hasattr(envs_mod.ScriptBrowserEnv.reset, "__wrapped__")

    def test_env_step_is_wrapped(self, instrument):
        import browser_env.envs as envs_mod

        assert hasattr(envs_mod.ScriptBrowserEnv.step, "__wrapped__")

    def test_env_close_is_wrapped(self, instrument):
        import browser_env.envs as envs_mod

        assert hasattr(envs_mod.ScriptBrowserEnv.close, "__wrapped__")

    def test_construct_agent_is_wrapped(self, instrument):
        import agent.agent as agent_mod

        assert hasattr(agent_mod.construct_agent, "__wrapped__")

    def test_next_action_is_wrapped(self, instrument):
        import agent.agent as agent_mod

        assert hasattr(agent_mod.PromptAgent.next_action, "__wrapped__")

    def test_prompt_constructors_wrapped(self, instrument):
        import agent.prompts.prompt_constructor as pc_mod

        assert hasattr(pc_mod.DirectPromptConstructor.construct, "__wrapped__")
        assert hasattr(pc_mod.CoTPromptConstructor.construct, "__wrapped__")

    def test_hf_completion_is_wrapped(self, instrument):
        import llms.providers.hf_utils as hf_mod

        assert hasattr(
            hf_mod.generate_from_huggingface_completion, "__wrapped__"
        )


class TestUnwrapping:
    """Verify that uninstrument restores original functions."""

    def test_env_reset_unwrapped(self, tracer_provider):
        instrumentor = WebarenaInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )
        import browser_env.envs as envs_mod

        assert hasattr(envs_mod.ScriptBrowserEnv.reset, "__wrapped__")
        instrumentor.uninstrument()
        assert not hasattr(envs_mod.ScriptBrowserEnv.reset, "__wrapped__")

    def test_construct_agent_unwrapped(self, tracer_provider):
        instrumentor = WebarenaInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )
        import agent.agent as agent_mod

        assert hasattr(agent_mod.construct_agent, "__wrapped__")
        instrumentor.uninstrument()
        assert not hasattr(agent_mod.construct_agent, "__wrapped__")

    def test_hf_unwrapped(self, tracer_provider):
        instrumentor = WebarenaInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider, skip_dep_check=True
        )
        import llms.providers.hf_utils as hf_mod

        assert hasattr(
            hf_mod.generate_from_huggingface_completion, "__wrapped__"
        )
        instrumentor.uninstrument()
        assert not hasattr(
            hf_mod.generate_from_huggingface_completion, "__wrapped__"
        )


class TestPartialPatchFailure:
    """Verify that individual patch failures are logged, not raised."""

    def test_missing_module_is_logged_not_raised(self, tracer_provider):
        """If a target module is not importable, instrument should still succeed
        for the other targets (just log warnings)."""
        # Remove one module temporarily
        saved = sys.modules.pop("llms.providers.hf_utils", None)
        saved_parent = sys.modules.get("llms.providers")
        if saved_parent and hasattr(saved_parent, "hf_utils"):
            delattr(saved_parent, "hf_utils")
        try:
            instrumentor = WebarenaInstrumentor()
            # Should not raise even though HF module is missing
            instrumentor.instrument(
                tracer_provider=tracer_provider, skip_dep_check=True
            )
            # HF should not be patched
            assert not instrumentor._patched_hf
            instrumentor.uninstrument()
        finally:
            # Restore
            if saved is not None:
                sys.modules["llms.providers.hf_utils"] = saved
                if saved_parent:
                    saved_parent.hf_utils = saved
