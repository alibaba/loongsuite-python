# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import unittest
from unittest.mock import patch

from google.genai import types as genai_types

from opentelemetry._logs import get_logger_provider
from opentelemetry.instrumentation.google_genai import tool_call_wrapper
from opentelemetry.instrumentation.google_genai._compat import TelemetryHandler
from opentelemetry.metrics import get_meter_provider
from opentelemetry.trace import get_tracer_provider

from ..common import otel_mocker


class TestCase(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(
            "os.environ",
            {
                "OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental",
            },
        )
        self.env_patcher.start()
        self._otel = otel_mocker.OTelMocker()
        self._otel.install()
        self._otel_wrapper = TelemetryHandler(
            tracer_provider=get_tracer_provider(),
            logger_provider=get_logger_provider(),
            meter_provider=get_meter_provider(),
        )

    def tearDown(self):
        self._otel.uninstall()
        self.env_patcher.stop()

    @property
    def otel(self):
        return self._otel

    @property
    def otel_wrapper(self):
        return self._otel_wrapper

    def wrap(self, tool_or_tools):
        return tool_call_wrapper.wrapped_tool(tool_or_tools, self.otel_wrapper)

    def test_wraps_none(self):
        result = self.wrap(None)
        self.assertIsNone(result)

    def test_wraps_multiple_tool_functions_as_list(self):
        def somefunction():
            pass

        def otherfunction():
            pass

        wrapped_functions = self.wrap([somefunction, otherfunction])
        wrapped_somefunction = wrapped_functions[0]
        wrapped_otherfunction = wrapped_functions[1]
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        somefunction()
        otherfunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_otherfunction()
        self.otel.assert_has_span_named("execute_tool otherfunction")

    def test_wraps_multiple_tool_functions_as_dict(self):
        def somefunction():
            pass

        def otherfunction():
            pass

        wrapped_functions = self.wrap(
            {"somefunction": somefunction, "otherfunction": otherfunction}
        )
        wrapped_somefunction = wrapped_functions["somefunction"]
        wrapped_otherfunction = wrapped_functions["otherfunction"]
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        somefunction()
        otherfunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_otherfunction()
        self.otel.assert_has_span_named("execute_tool otherfunction")

    def test_wraps_async_tool_function(self):
        async def somefunction():
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        asyncio.run(somefunction())
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        asyncio.run(wrapped_somefunction())
        self.otel.assert_has_span_named("execute_tool somefunction")

    def test_preserves_tool_dict(self):
        tool_dict = genai_types.ToolDict()
        wrapped_tool_dict = self.wrap(tool_dict)
        self.assertEqual(tool_dict, wrapped_tool_dict)

    def test_does_not_have_description_if_no_doc_string(self):
        def somefunction():
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        self.assertNotIn("gen_ai.tool.description", span.attributes)

    def test_has_description_if_doc_string_present(self):
        def somefunction():
            """An example tool call function."""

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        self.assertEqual(
            span.attributes["gen_ai.tool.description"],
            "An example tool call function.",
        )

    # Capture content must be enabled to get arguments
    @patch.dict(
        "os.environ",
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_AND_EVENT",
        },
    )
    def test_handles_various_arg_types(self):
        def somefunction(
            primitive_int=None,
            dict_arg=None,
            list_arg=None,
            heterogenous_list_arg=None,
        ):
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction(12345)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction(12345, {"key": "value"}, [1, 2, 3], [123, "abc"])
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        arguments = json.loads(span.attributes["gen_ai.tool.call.arguments"])
        self.assertEqual(
            arguments["code.function.parameters.primitive_int.type"], "int"
        )
        self.assertEqual(span.attributes["gen_ai.tool.name"], "somefunction")
        self.assertEqual(
            arguments["code.function.parameters.primitive_int.value"], 12345
        )
        self.assertEqual(
            arguments["code.function.parameters.dict_arg.type"], "dict"
        )
        self.assertEqual(
            arguments["code.function.parameters.dict_arg.value"],
            {"key": "value"},
        )
        self.assertEqual(
            arguments["code.function.parameters.list_arg.type"], "list"
        )
        self.assertEqual(
            arguments["code.function.parameters.list_arg.value"], [1, 2, 3]
        )
        self.assertEqual(
            arguments["code.function.parameters.heterogenous_list_arg.type"],
            "list",
        )
        self.assertEqual(
            arguments["code.function.parameters.heterogenous_list_arg.value"],
            [123, "abc"],
        )

    @patch.dict(
        "os.environ",
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT",
        },
    )
    def test_with_capture_content_disabled(self):
        def somefunction(arg=None):
            return arg

        wrapped_somefunction = self.wrap(somefunction)
        wrapped_somefunction("a string value")
        span = self.otel.get_span_named("execute_tool somefunction")

        self.assertNotIn(
            "gen_ai.tool.call.arguments",
            span.attributes,
        )
        self.assertNotIn(
            "gen_ai.tool.call.result",
            span.attributes,
        )

    def test_function_that_throws_exception(self):
        def somefunction(arg=None):
            raise Exception("Something went wrong")

        wrapped_somefunction = self.wrap(somefunction)
        try:
            wrapped_somefunction(12345)
        except Exception:
            span = self.otel.get_span_named("execute_tool somefunction")
            self.assertEqual(span.attributes["error.type"], "Exception")
