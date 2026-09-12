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

"""Smoke tests for ``BFCLv4Instrumentor``.

These tests do not require ``bfcl-eval`` to be installed; they only verify
that importing the package and calling ``instrument()`` / ``uninstrument()``
works (and degrades gracefully when ``bfcl-eval`` is missing).
"""

import importlib

import pytest


def test_import_instrumentor_package():
    module = importlib.import_module("opentelemetry.instrumentation.bfclv4")
    assert hasattr(module, "BFCLv4Instrumentor")


def test_instrumentation_dependencies_listed():
    from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor
    from opentelemetry.instrumentation.bfclv4.package import _instruments

    instr = BFCLv4Instrumentor()
    assert tuple(instr.instrumentation_dependencies()) == _instruments


def test_instrument_uninstrument_no_bfcl_no_raise():
    """When ``bfcl-eval`` is missing, every wrap call logs and continues.

    The instrumentor must not raise from ``instrument()`` /
    ``uninstrument()`` even if the target framework cannot be imported.
    """

    pytest.importorskip("opentelemetry.util.genai.extended_handler")
    from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

    instr = BFCLv4Instrumentor()
    instr.instrument()
    instr.uninstrument()
