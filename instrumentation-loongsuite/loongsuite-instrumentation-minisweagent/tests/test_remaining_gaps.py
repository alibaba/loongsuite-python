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

"""Tests for remaining coverage gaps in __init__.py, conversation.py, cli_wrappers.py, delegates.py.

Targets:
- __init__.py error-handling paths in _instrument/_uninstrument
- conversation.py: try_fill_entry_payload_from_mini_trajectory and error paths
- cli_wrappers.py: _hydrate_entry success, patch edge cases, unpatch error path
- delegates.py: _sanitize_tool_result error/fallback branches
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

# =====================================================================
# __init__.py  -- error-handling branches in _instrument / _uninstrument
# =====================================================================


class TestInstrumentErrorPaths:
    """Cover the except blocks in MiniSweAgentInstrumentor._instrument."""

    def test_get_environment_wrap_failure(self):
        """Lines 105-106: import minisweagent.environments fails."""
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        # Break the environments import
        saved = sys.modules.pop("minisweagent.environments", None)
        mini = sys.modules["minisweagent"]
        saved_attr = getattr(mini, "environments", None)
        if hasattr(mini, "environments"):
            delattr(mini, "environments")
        try:
            inst = MiniSweAgentInstrumentor()
            inst._instrument()  # should warn and continue
            inst._uninstrument()
        finally:
            if saved is not None:
                sys.modules["minisweagent.environments"] = saved
            if saved_attr is not None:
                setattr(mini, "environments", saved_attr)

    def test_patch_cli_app_failure(self):
        """Lines 110-111: patch_mini_cli_app_module() raises."""
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        with patch(
            "opentelemetry.instrumentation.minisweagent.internal.cli_wrappers.patch_mini_cli_app_module",
            side_effect=Exception("patch fail"),
        ):
            inst = MiniSweAgentInstrumentor()
            inst._instrument()  # should warn and continue
            inst._uninstrument()

    def test_wrap_function_wrapper_failure(self):
        """Lines 120-121 and 129-130: wrap_function_wrapper raises for run/step."""
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        with patch(
            "opentelemetry.instrumentation.minisweagent.wrap_function_wrapper",
            side_effect=Exception("wrap fail"),
        ):
            inst = MiniSweAgentInstrumentor()
            inst._instrument()  # should warn for both run and step, then continue
            inst._uninstrument()


class TestUninstrumentErrorPaths:
    """Cover the except blocks in MiniSweAgentInstrumentor._uninstrument."""

    def test_unwrap_agent_failure(self):
        """Lines 141-142: import DefaultAgent fails in _uninstrument."""
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        inst = MiniSweAgentInstrumentor()
        inst._instrument()

        # Break the agents.default import so uninstrument fails there
        saved = sys.modules.pop("minisweagent.agents.default", None)
        agents = sys.modules.get("minisweagent.agents")
        saved_attr = getattr(agents, "default", None) if agents else None
        if agents and hasattr(agents, "default"):
            delattr(agents, "default")
        try:
            inst._uninstrument()  # should catch and continue
        finally:
            if saved is not None:
                sys.modules["minisweagent.agents.default"] = saved
            if agents and saved_attr is not None:
                setattr(agents, "default", saved_attr)

    def test_unpatch_cli_failure(self):
        """Lines 150-151: unpatch_mini_cli_app_module raises."""
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        inst = MiniSweAgentInstrumentor()
        inst._instrument()

        with patch(
            "opentelemetry.instrumentation.minisweagent.internal.cli_wrappers.unpatch_mini_cli_app_module",
            side_effect=Exception("unpatch fail"),
        ):
            inst._uninstrument()  # should catch and continue

    def test_restore_get_environment_failure(self):
        """Lines 160-161: restoring get_environment fails in _uninstrument."""
        from opentelemetry.instrumentation.minisweagent import (
            MiniSweAgentInstrumentor,
        )

        inst = MiniSweAgentInstrumentor()
        inst._instrument()

        # Break the environments import for uninstrument's restore block
        saved = sys.modules.pop("minisweagent.environments", None)
        mini = sys.modules["minisweagent"]
        saved_attr = getattr(mini, "environments", None)
        if hasattr(mini, "environments"):
            delattr(mini, "environments")
        try:
            inst._uninstrument()  # should catch and continue
        finally:
            if saved is not None:
                sys.modules["minisweagent.environments"] = saved
            if saved_attr is not None:
                setattr(mini, "environments", saved_attr)
            # Reset the class-level cache since uninstrument might not have cleared it
            MiniSweAgentInstrumentor._original_get_environment = None


# =====================================================================
# conversation.py  -- try_fill_entry_payload_from_mini_trajectory
# =====================================================================


class TestTryFillEntryPayload:
    """Cover try_fill_entry_payload_from_mini_trajectory (lines 192-214)."""

    def _conv(self):
        from opentelemetry.instrumentation.minisweagent.internal import (
            conversation,
        )

        return conversation

    def test_no_global_config_dir_returns_none(self):
        """When global_config_dir is not importable, return None."""
        result = self._conv().try_fill_entry_payload_from_mini_trajectory()
        assert result is None

    def test_file_not_found_returns_none(self, tmp_path):
        """When the trajectory file does not exist, return None."""
        mini = sys.modules["minisweagent"]
        mini.global_config_dir = str(tmp_path)
        try:
            result = self._conv().try_fill_entry_payload_from_mini_trajectory()
            assert result is None
        finally:
            if hasattr(mini, "global_config_dir"):
                delattr(mini, "global_config_dir")

    def test_file_too_large_returns_none(self, tmp_path):
        """When the trajectory file exceeds size limit, return None."""
        mini = sys.modules["minisweagent"]
        mini.global_config_dir = str(tmp_path)
        traj_file = tmp_path / "last_mini_run.traj.json"
        # Write more than 8MB
        traj_file.write_text("x" * (8_000_001), encoding="utf-8")
        try:
            result = self._conv().try_fill_entry_payload_from_mini_trajectory()
            assert result is None
        finally:
            if hasattr(mini, "global_config_dir"):
                delattr(mini, "global_config_dir")

    def test_invalid_json_returns_none(self, tmp_path):
        """When the trajectory file has invalid JSON, return None."""
        mini = sys.modules["minisweagent"]
        mini.global_config_dir = str(tmp_path)
        traj_file = tmp_path / "last_mini_run.traj.json"
        traj_file.write_text("not valid json {{{", encoding="utf-8")
        try:
            result = self._conv().try_fill_entry_payload_from_mini_trajectory()
            assert result is None
        finally:
            if hasattr(mini, "global_config_dir"):
                delattr(mini, "global_config_dir")

    def test_no_messages_key_returns_none(self, tmp_path):
        """When JSON has no 'messages' key, return None."""
        mini = sys.modules["minisweagent"]
        mini.global_config_dir = str(tmp_path)
        traj_file = tmp_path / "last_mini_run.traj.json"
        traj_file.write_text(json.dumps({"other": "data"}), encoding="utf-8")
        try:
            result = self._conv().try_fill_entry_payload_from_mini_trajectory()
            assert result is None
        finally:
            if hasattr(mini, "global_config_dir"):
                delattr(mini, "global_config_dir")

    def test_messages_not_list_returns_none(self, tmp_path):
        """When 'messages' is not a list, return None."""
        mini = sys.modules["minisweagent"]
        mini.global_config_dir = str(tmp_path)
        traj_file = tmp_path / "last_mini_run.traj.json"
        traj_file.write_text(
            json.dumps({"messages": "not a list"}), encoding="utf-8"
        )
        try:
            result = self._conv().try_fill_entry_payload_from_mini_trajectory()
            assert result is None
        finally:
            if hasattr(mini, "global_config_dir"):
                delattr(mini, "global_config_dir")

    def test_messages_with_no_dicts_returns_none(self, tmp_path):
        """When messages list has no dict entries, return None."""
        mini = sys.modules["minisweagent"]
        mini.global_config_dir = str(tmp_path)
        traj_file = tmp_path / "last_mini_run.traj.json"
        traj_file.write_text(
            json.dumps({"messages": ["str", 42, None]}), encoding="utf-8"
        )
        try:
            result = self._conv().try_fill_entry_payload_from_mini_trajectory()
            assert result is None
        finally:
            if hasattr(mini, "global_config_dir"):
                delattr(mini, "global_config_dir")

    def test_valid_trajectory_returns_payload(self, tmp_path):
        """When the trajectory file is valid, return a payload dict."""
        mini = sys.modules["minisweagent"]
        mini.global_config_dir = str(tmp_path)
        traj_file = tmp_path / "last_mini_run.traj.json"
        traj_data = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        }
        traj_file.write_text(json.dumps(traj_data), encoding="utf-8")
        try:
            result = self._conv().try_fill_entry_payload_from_mini_trajectory()
            assert result is not None
            assert "system_instruction" in result
            assert "input_messages" in result
            assert "output_messages" in result
            assert "tool_definitions" in result
        finally:
            if hasattr(mini, "global_config_dir"):
                delattr(mini, "global_config_dir")

    def test_build_payload_exception_returns_none(self, tmp_path):
        """When build_invoke_payload_from_messages raises, return None."""
        mini = sys.modules["minisweagent"]
        mini.global_config_dir = str(tmp_path)
        traj_file = tmp_path / "last_mini_run.traj.json"
        traj_data = {"messages": [{"role": "user", "content": "hello"}]}
        traj_file.write_text(json.dumps(traj_data), encoding="utf-8")
        try:
            conversation = self._conv()
            with patch.object(
                conversation,
                "build_invoke_payload_from_messages",
                side_effect=Exception("build failed"),
            ) as mock_build:
                result = (
                    conversation.try_fill_entry_payload_from_mini_trajectory()
                )
                assert result is None
                mock_build.assert_called_once()
        finally:
            if hasattr(mini, "global_config_dir"):
                delattr(mini, "global_config_dir")


class TestConversationSerializationError:
    """Cover the except block in build_invoke_payload_from_messages (lines 167-168)."""

    def test_message_conversion_failure_caught(self):
        from opentelemetry.instrumentation.minisweagent.internal import (
            conversation,
        )

        with patch.object(
            conversation,
            "_message_to_semconv_messages",
            side_effect=Exception("conversion error"),
        ):
            payload = conversation.build_invoke_payload_from_messages(
                [
                    {"role": "user", "content": "hello"},
                ]
            )
            # Exception caught; partial results
            assert payload["system_instruction"] == []
            assert "tool_definitions" in payload


# =====================================================================
# cli_wrappers.py  -- _hydrate_entry success, patch edge cases, unpatch error
# =====================================================================


class TestHydrateEntrySuccess:
    """Cover _hydrate_entry success path (line 35) and error path (lines 36-37)."""

    def test_hydrate_applies_payload(self):
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            _MiniTyperAppProxy,
        )

        inner = MagicMock(return_value="result")
        proxy = _MiniTyperAppProxy(inner)

        fake_payload = {
            "input_messages": ["im"],
            "output_messages": ["om"],
            "system_instruction": ["si"],
            "tool_definitions": ["td"],
        }

        with patch(
            "opentelemetry.instrumentation.minisweagent.internal.cli_wrappers.try_fill_entry_payload_from_mini_trajectory",
            return_value=fake_payload,
        ):
            result = proxy()

        assert result == "result"

    def test_hydrate_exception_suppressed(self):
        """Lines 36-37: exception inside _hydrate_entry try block is caught."""
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            _MiniTyperAppProxy,
        )

        inner = MagicMock(return_value="result")
        proxy = _MiniTyperAppProxy(inner)

        with patch(
            "opentelemetry.instrumentation.minisweagent.internal.cli_wrappers.try_fill_entry_payload_from_mini_trajectory",
            side_effect=Exception("hydrate boom"),
        ):
            result = proxy()

        assert result == "result"


class TestPatchCliEdgeCases:
    """Cover patch_mini_cli_app_module edge cases."""

    def test_import_failure_returns_early(self):
        """Lines 78-82: when run.mini cannot be imported."""
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            patch_mini_cli_app_module,
        )

        saved = sys.modules.pop("minisweagent.run.mini", None)
        run_pkg = sys.modules.get("minisweagent.run")
        saved_attr = getattr(run_pkg, "mini", None) if run_pkg else None
        if run_pkg and hasattr(run_pkg, "mini"):
            delattr(run_pkg, "mini")
        try:
            patch_mini_cli_app_module()  # should catch ImportError and return
        finally:
            if saved is not None:
                sys.modules["minisweagent.run.mini"] = saved
            if run_pkg and saved_attr is not None:
                setattr(run_pkg, "mini", saved_attr)

    def test_app_is_none_returns_early(self):
        """Line 89: when mini_mod.app is None."""
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            _PATCH_FLAG,
            patch_mini_cli_app_module,
        )

        mini_mod = sys.modules["minisweagent.run.mini"]
        original_app = mini_mod.app
        mini_mod.app = None
        try:
            patch_mini_cli_app_module()  # inner is None, returns early
            assert not getattr(mini_mod, _PATCH_FLAG, False)
        finally:
            mini_mod.app = original_app

    def test_app_already_proxy_returns_early(self):
        """Line 89: when mini_mod.app is already a _MiniTyperAppProxy."""
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            _PATCH_FLAG,
            _MiniTyperAppProxy,
            patch_mini_cli_app_module,
            unpatch_mini_cli_app_module,
        )

        # First patch normally
        patch_mini_cli_app_module()
        mini_mod = sys.modules["minisweagent.run.mini"]
        # Manually remove patch flag but keep proxy as app
        # so the isinstance check triggers on re-patch
        if hasattr(mini_mod, _PATCH_FLAG):
            delattr(mini_mod, _PATCH_FLAG)
        proxy = mini_mod.app
        assert isinstance(proxy, _MiniTyperAppProxy)

        # Second patch should detect proxy and return early
        patch_mini_cli_app_module()

        unpatch_mini_cli_app_module()


class TestUnpatchCliError:
    """Cover unpatch_mini_cli_app_module error path (lines 105-106)."""

    def test_delattr_failure_caught(self):
        from opentelemetry.instrumentation.minisweagent.internal.cli_wrappers import (
            _ORIG_APP_ATTR,
            patch_mini_cli_app_module,
            unpatch_mini_cli_app_module,
        )

        patch_mini_cli_app_module()
        mini_mod = sys.modules["minisweagent.run.mini"]

        # Remove _ORIG_APP_ATTR so delattr in unpatch will raise AttributeError
        if hasattr(mini_mod, _ORIG_APP_ATTR):
            delattr(mini_mod, _ORIG_APP_ATTR)

        # unpatch should catch the AttributeError from the second delattr
        unpatch_mini_cli_app_module()  # should not raise


# =====================================================================
# delegates.py  -- _sanitize_tool_result error branches (lines 21-26)
# =====================================================================


class TestSanitizeToolResultErrorBranches:
    """Cover the exception fallback branches in _sanitize_tool_result."""

    def test_repr_fallback_on_json_failure(self):
        """Lines 21-24: json.dumps fails, falls back to repr."""
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            _sanitize_tool_result,
        )

        class BadStr:
            def __str__(self):
                raise TypeError("cannot stringify")

        result = _sanitize_tool_result({"key": BadStr()})
        assert "repr" in result
        assert isinstance(result["repr"], str)

    def test_total_failure_returns_error_dict(self):
        """Lines 25-26: both json.dumps and repr fail."""
        from opentelemetry.instrumentation.minisweagent.internal.delegates import (
            _sanitize_tool_result,
        )

        class TotallyBad:
            def __str__(self):
                raise TypeError("no str")

            def __repr__(self):
                raise RuntimeError("no repr either")

        # repr(payload) where payload contains TotallyBad will fail
        # because dict.__repr__ calls repr on values
        result = _sanitize_tool_result({"key": TotallyBad()})
        assert result == {"error": "unserializable_tool_result"}
