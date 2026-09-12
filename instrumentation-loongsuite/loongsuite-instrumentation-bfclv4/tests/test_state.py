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

"""Tests for ``internal/state.py`` -- uncovered edge cases."""

from __future__ import annotations


class TestStateFunctions:
    def test_reset_state_normal(self):
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            get_state,
            init_state,
            reset_state,
        )

        token = init_state()
        assert get_state() is not None
        reset_state(token)
        assert get_state() is None

    def test_bump_round_no_state(self):
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            bump_round,
            get_state,
        )

        # Ensure state is not set
        assert get_state() is None
        result = bump_round()
        assert result == 1

    def test_reset_round_for_turn(self):
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            bump_round,
            get_state,
            init_state,
            reset_round_for_turn,
            reset_state,
        )

        token = init_state()
        try:
            bump_round()
            bump_round()
            assert get_state()["fc_round"] == 2
            reset_round_for_turn()
            assert get_state()["fc_round"] == 0
        finally:
            reset_state(token)

    def test_reset_round_for_turn_no_state(self):
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            get_state,
            reset_round_for_turn,
        )

        assert get_state() is None
        # Should not raise
        reset_round_for_turn()

    def test_bump_turn_no_state(self):
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            bump_turn,
            get_state,
        )

        assert get_state() is None
        result = bump_turn()
        assert result == 0

    def test_next_tool_index_no_state(self):
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            get_state,
            next_tool_index,
        )

        assert get_state() is None
        result = next_tool_index()
        assert result == 0

    def test_bump_turn_resets_fc_round(self):
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            bump_round,
            bump_turn,
            get_state,
            init_state,
            reset_state,
        )

        token = init_state()
        try:
            bump_round()
            bump_round()
            assert get_state()["fc_round"] == 2
            bump_turn()
            assert get_state()["fc_round"] == 0
            assert get_state()["turn_idx"] == 1
        finally:
            reset_state(token)

    def test_next_tool_index_increments(self):
        from opentelemetry.instrumentation.bfclv4.internal.state import (
            get_state,
            init_state,
            next_tool_index,
            reset_state,
        )

        token = init_state()
        try:
            assert next_tool_index() == 0
            assert next_tool_index() == 1
            assert next_tool_index() == 2
            assert get_state()["tool_index"] == 3
        finally:
            reset_state(token)
