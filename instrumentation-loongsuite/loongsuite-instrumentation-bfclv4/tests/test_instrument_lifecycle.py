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

"""Tests for ``BFCLv4Instrumentor._instrument()`` / ``_uninstrument()``
lifecycle with mock bfcl_eval modules available."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


class TestIterHandlerClasses:
    def test_iter_handler_classes_returns_classes(self):
        from opentelemetry.instrumentation.bfclv4 import _iter_handler_classes

        classes = _iter_handler_classes()
        assert isinstance(classes, list)
        # Our mock MODEL_CONFIG_MAPPING has one entry
        assert len(classes) >= 1

    def test_iter_handler_classes_deduplicates(self):
        from opentelemetry.instrumentation.bfclv4 import _iter_handler_classes

        class _Handler:
            pass

        class _Cfg:
            model_handler = _Handler

        model_config = sys.modules["bfcl_eval.constants.model_config"]
        model_config.MODEL_CONFIG_MAPPING = {
            "a": _Cfg(),
            "b": _Cfg(),
        }

        classes = _iter_handler_classes()
        assert len(classes) == 1

    def test_iter_handler_classes_skips_non_type(self):
        from opentelemetry.instrumentation.bfclv4 import _iter_handler_classes

        model_config = sys.modules["bfcl_eval.constants.model_config"]

        class _BadConfig:
            model_handler = "not a class"

        model_config.MODEL_CONFIG_MAPPING = {"bad": _BadConfig()}
        classes = _iter_handler_classes()
        assert len(classes) == 0

    def test_iter_handler_classes_skips_none_handler(self):
        from opentelemetry.instrumentation.bfclv4 import _iter_handler_classes

        model_config = sys.modules["bfcl_eval.constants.model_config"]

        class _NoHandler:
            pass

        model_config.MODEL_CONFIG_MAPPING = {"nh": _NoHandler()}
        classes = _iter_handler_classes()
        assert len(classes) == 0

    def test_iter_handler_classes_import_error(self):
        from opentelemetry.instrumentation.bfclv4 import _iter_handler_classes

        # Remove the mock module to simulate import error
        saved = sys.modules.pop("bfcl_eval.constants.model_config")
        try:
            classes = _iter_handler_classes()
            assert classes == []
        finally:
            sys.modules["bfcl_eval.constants.model_config"] = saved


class TestInstrumentUninstrument:
    def test_instrument_wraps_generate_results(self):
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()
        instr.instrument(skip_dep_check=True)
        try:
            assert instr._entry_wrapped is True
        finally:
            instr.uninstrument()

    def test_instrument_wraps_base_handler_inference(self):
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()
        instr.instrument(skip_dep_check=True)
        try:
            assert instr._inference_wrapped is True
        finally:
            instr.uninstrument()

    def test_instrument_wraps_tool_targets(self):
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()
        instr.instrument(skip_dep_check=True)
        try:
            assert instr._tool_wrapped is True
            assert len(instr._tool_targets) >= 1
        finally:
            instr.uninstrument()

    def test_instrument_wraps_handler_methods(self):
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()
        instr.instrument(skip_dep_check=True)
        try:
            assert len(instr._wrapped_query_methods) >= 1
        finally:
            instr.uninstrument()

    def test_uninstrument_clears_state(self):
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()
        instr.instrument(skip_dep_check=True)
        instr.uninstrument()

        assert instr._entry_wrapped is False
        assert instr._inference_wrapped is False
        assert instr._tool_wrapped is False
        assert instr._wrapped_query_methods == []
        assert instr._wrapped_turn_methods == []
        assert instr._tool_targets == []

    def test_instrument_twice_uninstrument_once(self):
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()
        instr.instrument(skip_dep_check=True)
        instr.uninstrument()
        # Second instrument/uninstrument should work
        instr.instrument(skip_dep_check=True)
        instr.uninstrument()

    def test_uninstrument_before_instrument(self):
        """Uninstrumenting without prior instrument should not crash."""
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()
        # Reset manually as BaseInstrumentor tracks state
        instr._entry_wrapped = False
        instr._inference_wrapped = False
        instr._tool_wrapped = False
        instr._tool_targets = []
        instr._wrapped_query_methods = []
        instr._wrapped_turn_methods = []
        # Call _uninstrument directly (bypassing BaseInstrumentor check)
        instr._uninstrument()

    def test_instrument_handler_method_wrap_failure(self):
        """If wrapping a handler method fails, it should log and continue."""
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()

        # Make handler class's module invalid to trigger wrap failure
        base_handler = sys.modules["bfcl_eval.model_handler.base_handler"]

        class _BadHandler(base_handler.BaseHandler):
            def _query_FC(self, *a, **kw):
                return "x", 0.1

        _BadHandler.__module__ = "nonexistent.module.path"
        _BadHandler.__name__ = "BadHandler"

        class _Cfg:
            model_handler = _BadHandler

        model_config = sys.modules["bfcl_eval.constants.model_config"]
        model_config.MODEL_CONFIG_MAPPING = {
            "bad": _Cfg(),
        }

        # Should not raise
        instr.instrument(skip_dep_check=True)
        instr.uninstrument()

    def test_uninstrument_with_unwrap_failure(self):
        """If unwrapping fails, it should log and continue."""
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()
        instr.instrument(skip_dep_check=True)

        # Corrupt the tool_targets to trigger unwrap failure
        instr._tool_targets.append(("nonexistent.module.that.fails", "fn"))

        # Should not raise
        instr.uninstrument()

    def test_instrument_with_turn_methods(self):
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()
        instr.instrument(skip_dep_check=True)
        try:
            # Our mock handler has turn methods
            assert len(instr._wrapped_turn_methods) >= 1
        finally:
            instr.uninstrument()


class TestInstrumentWrapFailures:
    """Test the exception-handling paths when wrap_function_wrapper fails."""

    def test_entry_wrap_failure_logs_warning(self):
        """When generate_results module is broken, wrapping should log and continue."""
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        # Remove the generate_results function to cause wrapping to fail
        gen_mod = sys.modules["bfcl_eval._llm_response_generation"]
        saved = gen_mod.generate_results
        del gen_mod.generate_results
        try:
            instr = BFCLv4Instrumentor()
            instr.instrument(skip_dep_check=True)
            assert instr._entry_wrapped is False
            instr.uninstrument()
        finally:
            gen_mod.generate_results = saved

    def test_agent_wrap_failure_logs_warning(self):
        """When BaseHandler.inference is broken, wrapping should log and continue."""
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        base_mod = sys.modules["bfcl_eval.model_handler.base_handler"]
        saved = base_mod.BaseHandler
        # Replace with something that can't be wrapped
        base_mod.BaseHandler = "not_a_class"
        try:
            instr = BFCLv4Instrumentor()
            instr.instrument(skip_dep_check=True)
            assert instr._inference_wrapped is False
            instr.uninstrument()
        finally:
            base_mod.BaseHandler = saved

    def test_tool_wrap_failure_logs_debug(self):
        """When tool target doesn't exist, wrapping should log and continue."""
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        base_mod = sys.modules["bfcl_eval.model_handler.base_handler"]
        saved = getattr(base_mod, "execute_multi_turn_func_call", None)
        if hasattr(base_mod, "execute_multi_turn_func_call"):
            del base_mod.execute_multi_turn_func_call
        try:
            instr = BFCLv4Instrumentor()
            instr.instrument(skip_dep_check=True)
            assert instr._tool_wrapped is False
            instr.uninstrument()
        finally:
            if saved is not None:
                base_mod.execute_multi_turn_func_call = saved

    def test_uninstrument_query_unwrap_failure(self):
        """If unwrapping query methods fails, should log and continue."""
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()
        instr.instrument(skip_dep_check=True)

        # Corrupt the wrapped_query_methods with a bad class
        class _Fake:
            pass

        instr._wrapped_query_methods.append((_Fake, "nonexistent"))

        # Should not raise
        instr.uninstrument()

    def test_uninstrument_turn_unwrap_failure(self):
        """If unwrapping turn methods fails, should log and continue."""
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()
        instr.instrument(skip_dep_check=True)

        class _Fake:
            pass

        instr._wrapped_turn_methods.append((_Fake, "nonexistent"))
        instr.uninstrument()

    def test_uninstrument_inference_unwrap_failure(self):
        """If unwrapping BaseHandler.inference fails, should log and continue."""
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()
        instr.instrument(skip_dep_check=True)

        # Corrupt the base handler module to make unwrap fail
        base_mod = sys.modules["bfcl_eval.model_handler.base_handler"]
        saved = base_mod.BaseHandler
        base_mod.BaseHandler = MagicMock()
        try:
            instr.uninstrument()
        finally:
            base_mod.BaseHandler = saved

    def test_uninstrument_entry_unwrap_failure(self):
        """If unwrapping generate_results fails, should log and continue."""
        from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

        instr = BFCLv4Instrumentor()
        instr.instrument(skip_dep_check=True)

        # Remove the generate_results to make unwrap fail
        gen_mod = sys.modules["bfcl_eval._llm_response_generation"]
        saved = gen_mod.generate_results
        del gen_mod.generate_results
        try:
            instr.uninstrument()
        finally:
            gen_mod.generate_results = saved


class TestVersion:
    def test_version_string(self):
        from opentelemetry.instrumentation.bfclv4.version import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0
