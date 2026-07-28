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

"""Shared pytest hooks for multimodal upload tests."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

from collections.abc import Iterator

import pytest

from .multimodal_test_helpers import reset_multimodal_runtime_state_for_test


@pytest.fixture(autouse=True)
def _isolate_multimodal_runtime_state() -> Iterator[None]:
    """Reset singleton state before/after each test.

    Tests that patch ``os.environ`` must call
    ``reset_multimodal_runtime_state_for_test()`` again inside the patch
    so the runtime snapshot is rebuilt from the patched env.
    """
    reset_multimodal_runtime_state_for_test()
    yield
    reset_multimodal_runtime_state_for_test()
