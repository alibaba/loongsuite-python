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

"""Tests for opentelemetry.instrumentation.minisweagent.config."""

from __future__ import annotations

import os
from unittest.mock import patch


class TestIntEnv:
    """Tests for the ``_int_env`` helper."""

    def test_valid_integer_env(self):
        """When the env var holds a valid integer string, return it."""
        with patch.dict(os.environ, {"MY_TEST_VAR": "42"}):
            from opentelemetry.instrumentation.minisweagent.config import (
                _int_env,
            )

            assert _int_env("MY_TEST_VAR", "10") == 42

    def test_missing_env_uses_default(self):
        """When the env var is missing, return the parsed default."""
        env_key = "DOES_NOT_EXIST_TEST_VAR_12345"
        os.environ.pop(env_key, None)

        from opentelemetry.instrumentation.minisweagent.config import _int_env

        assert _int_env(env_key, "99") == 99

    def test_invalid_env_falls_back_to_default(self):
        """When the env var holds a non-integer string, fall back to the default."""
        with patch.dict(os.environ, {"MY_BAD_VAR": "not_a_number"}):
            from opentelemetry.instrumentation.minisweagent.config import (
                _int_env,
            )

            assert _int_env("MY_BAD_VAR", "7") == 7

    def test_empty_string_env_falls_back(self):
        """An empty string is not a valid int; fall back to default."""
        with patch.dict(os.environ, {"MY_EMPTY_VAR": ""}):
            from opentelemetry.instrumentation.minisweagent.config import (
                _int_env,
            )

            assert _int_env("MY_EMPTY_VAR", "5") == 5

    def test_negative_integer(self):
        """Negative integers should parse correctly."""
        with patch.dict(os.environ, {"MY_NEG_VAR": "-10"}):
            from opentelemetry.instrumentation.minisweagent.config import (
                _int_env,
            )

            assert _int_env("MY_NEG_VAR", "0") == -10

    def test_zero_value(self):
        """Zero should parse correctly."""
        with patch.dict(os.environ, {"MY_ZERO_VAR": "0"}):
            from opentelemetry.instrumentation.minisweagent.config import (
                _int_env,
            )

            assert _int_env("MY_ZERO_VAR", "100") == 0

    def test_whitespace_value_falls_back(self):
        """Whitespace-only value is not a valid int."""
        with patch.dict(os.environ, {"MY_WS_VAR": "  "}):
            from opentelemetry.instrumentation.minisweagent.config import (
                _int_env,
            )

            assert _int_env("MY_WS_VAR", "3") == 3

    def test_float_string_falls_back(self):
        """A float string like '3.14' is not a valid int."""
        with patch.dict(os.environ, {"MY_FLOAT_VAR": "3.14"}):
            from opentelemetry.instrumentation.minisweagent.config import (
                _int_env,
            )

            assert _int_env("MY_FLOAT_VAR", "9") == 9

    def test_large_integer(self):
        """Large integers should parse correctly."""
        with patch.dict(os.environ, {"MY_LARGE_VAR": "999999999"}):
            from opentelemetry.instrumentation.minisweagent.config import (
                _int_env,
            )

            assert _int_env("MY_LARGE_VAR", "0") == 999999999


class TestModuleLevelConstants:
    """Test the module-level constants.

    These are computed at import time using ``_int_env``.  We cannot
    ``importlib.reload`` because of namespace-package interactions,
    so instead we verify the defaults (env vars are not set during CI)
    and test ``_int_env`` for override/fallback semantics separately.
    """

    def test_task_preview_max_len_default_value(self):
        """The default should be 256 (or whatever the env var overrides to)."""
        from opentelemetry.instrumentation.minisweagent.config import (
            OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN,
            _int_env,
        )

        expected = _int_env("OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN", "256")
        assert OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN == expected

    def test_command_preview_max_len_default_value(self):
        """The default should be 256 (or whatever the env var overrides to)."""
        from opentelemetry.instrumentation.minisweagent.config import (
            OTEL_MINISWEAGENT_COMMAND_PREVIEW_MAX_LEN,
            _int_env,
        )

        expected = _int_env("OTEL_MINISWEAGENT_COMMAND_PREVIEW_MAX_LEN", "256")
        assert OTEL_MINISWEAGENT_COMMAND_PREVIEW_MAX_LEN == expected

    def test_task_preview_env_override_via_int_env(self):
        """Verify that ``_int_env`` respects the override for the task constant."""
        from opentelemetry.instrumentation.minisweagent.config import _int_env

        with patch.dict(
            os.environ, {"OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN": "512"}
        ):
            assert (
                _int_env("OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN", "256")
                == 512
            )

    def test_command_preview_env_override_via_int_env(self):
        """Verify that ``_int_env`` respects the override for the command constant."""
        from opentelemetry.instrumentation.minisweagent.config import _int_env

        with patch.dict(
            os.environ, {"OTEL_MINISWEAGENT_COMMAND_PREVIEW_MAX_LEN": "128"}
        ):
            assert (
                _int_env("OTEL_MINISWEAGENT_COMMAND_PREVIEW_MAX_LEN", "256")
                == 128
            )

    def test_task_preview_invalid_env_via_int_env(self):
        """Non-integer env var should fall back to 256."""
        from opentelemetry.instrumentation.minisweagent.config import _int_env

        with patch.dict(
            os.environ, {"OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN": "abc"}
        ):
            assert (
                _int_env("OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN", "256")
                == 256
            )

    def test_constants_are_int(self):
        """Module-level constants must be integers."""
        from opentelemetry.instrumentation.minisweagent.config import (
            OTEL_MINISWEAGENT_COMMAND_PREVIEW_MAX_LEN,
            OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN,
        )

        assert isinstance(OTEL_MINISWEAGENT_TASK_PREVIEW_MAX_LEN, int)
        assert isinstance(OTEL_MINISWEAGENT_COMMAND_PREVIEW_MAX_LEN, int)


class TestEntrySpanActive:
    """Tests for the ENTRY_SPAN_ACTIVE ContextVar."""

    def test_default_is_false(self):
        from opentelemetry.instrumentation.minisweagent.config import (
            ENTRY_SPAN_ACTIVE,
        )

        assert ENTRY_SPAN_ACTIVE.get() is False

    def test_set_and_reset(self):
        from opentelemetry.instrumentation.minisweagent.config import (
            ENTRY_SPAN_ACTIVE,
        )

        token = ENTRY_SPAN_ACTIVE.set(True)
        assert ENTRY_SPAN_ACTIVE.get() is True
        ENTRY_SPAN_ACTIVE.reset(token)
        assert ENTRY_SPAN_ACTIVE.get() is False

    def test_nested_set_and_reset(self):
        from opentelemetry.instrumentation.minisweagent.config import (
            ENTRY_SPAN_ACTIVE,
        )

        t1 = ENTRY_SPAN_ACTIVE.set(True)
        t2 = ENTRY_SPAN_ACTIVE.set(False)
        assert ENTRY_SPAN_ACTIVE.get() is False
        ENTRY_SPAN_ACTIVE.reset(t2)
        assert ENTRY_SPAN_ACTIVE.get() is True
        ENTRY_SPAN_ACTIVE.reset(t1)
        assert ENTRY_SPAN_ACTIVE.get() is False
