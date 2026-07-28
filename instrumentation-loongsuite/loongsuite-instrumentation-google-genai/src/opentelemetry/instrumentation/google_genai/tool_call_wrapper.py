# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import json
from typing import Any, Callable, Optional, Union

from google.genai.types import (
    ToolListUnion,
    ToolListUnionDict,
    ToolOrDict,
)

from ._compat import TelemetryHandler

ToolFunction = Callable[..., Any]


def _is_primitive(value):
    return isinstance(value, (str, int, bool, float))


def _to_otel_value(python_value):
    """Coerces parameters to something representable with Open Telemetry."""
    if python_value is None or _is_primitive(python_value):
        return python_value
    if isinstance(python_value, list):
        return [_to_otel_value(x) for x in python_value]
    if isinstance(python_value, dict):
        return {
            key: _to_otel_value(val) for (key, val) in python_value.items()
        }
    if hasattr(python_value, "model_dump"):
        return python_value.model_dump()
    if hasattr(python_value, "__dict__"):
        return _to_otel_value(python_value.__dict__)
    return repr(python_value)


# There is no canonical way to serialize a Python object to a span attribute value.
# Span attribute values currently must be one of the primitive types, or a homogeneous list of primitive types.
# In the future the value will be expanded to include None, heterogeneous lists of primitive types, and a Map of these types.
# See https://github.com/open-telemetry/opentelemetry-specification/pull/4485
def _get_function_args(wrapped_function, function_args, function_kwargs):
    """Records the details about a function invocation as span attributes."""
    function_arg_attr = {}
    signature = inspect.signature(wrapped_function)
    params = list(signature.parameters.values())
    for index, entry in enumerate(function_args):
        param_name = f"args[{index}]"
        if index < len(params):
            param_name = params[index].name
        function_arg_attr[f"code.function.parameters.{param_name}.type"] = (
            type(entry).__name__
        )
        function_arg_attr[f"code.function.parameters.{param_name}.value"] = (
            _to_otel_value(entry)
        )
    for key, value in function_kwargs.items():
        function_arg_attr[f"code.function.parameters.{key}.type"] = type(
            value
        ).__name__
        function_arg_attr[f"code.function.parameters.{key}.value"] = (
            _to_otel_value(value)
        )
    return function_arg_attr


def _wrap_tool_function(
    tool_function: ToolFunction,
    telemetry_handler: TelemetryHandler,
):
    if inspect.iscoroutinefunction(tool_function):

        @functools.wraps(tool_function)
        async def wrapped_function(*args, **kwargs):
            # Always json.dumps. First we convert args / result to something that we can serialize, then we serialize.
            # The return value of _to_otel_value could be a dict, which currently cannot be a span attribute..
            # In the future that could change (see https://github.com/open-telemetry/opentelemetry-specification/pull/4485), and we could possibly stop using json.dumps here.
            with telemetry_handler.tool(
                tool_function.__name__,
                tool_description=tool_function.__doc__,
            ) as tool_invocation:
                # Do this before calling the tool in case that crashes.
                if tool_invocation.should_capture_content_on_span:
                    tool_invocation.arguments = json.dumps(
                        _get_function_args(tool_function, args, kwargs)
                    )
                result = await tool_function(*args, **kwargs)
                if tool_invocation.should_capture_content_on_span:
                    tool_invocation.tool_result = json.dumps(
                        _to_otel_value(result)
                    )
            return result
    else:

        @functools.wraps(tool_function)
        def wrapped_function(*args, **kwargs):
            with telemetry_handler.tool(
                tool_function.__name__,
                tool_description=tool_function.__doc__,
            ) as tool_invocation:
                # Do this before calling the tool in case that crashes.
                if tool_invocation.should_capture_content_on_span:
                    tool_invocation.arguments = json.dumps(
                        _get_function_args(tool_function, args, kwargs)
                    )
                result = tool_function(*args, **kwargs)
                if tool_invocation.should_capture_content_on_span:
                    tool_invocation.tool_result = json.dumps(
                        _to_otel_value(result)
                    )
            return result

    return wrapped_function


def wrapped_tool(
    tool_or_tools: Optional[
        Union[ToolFunction, ToolOrDict, ToolListUnion, ToolListUnionDict]
    ],
    telemetry_handler: TelemetryHandler,
):
    if tool_or_tools is None:
        return None
    if isinstance(tool_or_tools, list):
        return [
            wrapped_tool(tool, telemetry_handler) for tool in tool_or_tools
        ]
    if isinstance(tool_or_tools, dict):
        return {
            key: wrapped_tool(tool, telemetry_handler)
            for (key, tool) in tool_or_tools.items()
        }
    if callable(tool_or_tools):
        return _wrap_tool_function(tool_or_tools, telemetry_handler)
    return tool_or_tools
