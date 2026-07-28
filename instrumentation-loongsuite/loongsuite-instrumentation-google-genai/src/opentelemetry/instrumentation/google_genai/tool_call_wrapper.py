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

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from google.genai.types import (
    ToolListUnion,
    ToolListUnionDict,
    ToolOrDict,
)

from opentelemetry.util.genai import hook_advice

from ._compat import TelemetryHandler, ToolInvocation

ToolFunction = Callable[..., Any]


@dataclass
class _ToolAdviceState:
    invocation: ToolInvocation


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


@hook_advice("google-genai", "prepare_tool")
def _prepare_tool_advice(
    tool_function: ToolFunction,
    telemetry_handler: TelemetryHandler,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> _ToolAdviceState:
    invocation = None
    try:
        invocation = telemetry_handler.tool(
            tool_function.__name__,
            tool_description=tool_function.__doc__,
        )
        if invocation.should_capture_content_on_span:
            invocation.arguments = json.dumps(
                _get_function_args(tool_function, args, kwargs)
            )
        return _ToolAdviceState(invocation=invocation)
    except Exception:
        if invocation is not None:
            telemetry_handler.abandon_tool(invocation)
        raise


@hook_advice("google-genai", "complete_tool")
def _complete_tool_advice(
    state: _ToolAdviceState,
    result: Any,
) -> None:
    try:
        if state.invocation.should_capture_content_on_span:
            state.invocation.tool_result = json.dumps(_to_otel_value(result))
    finally:
        state.invocation.stop()


@hook_advice("google-genai", "fail_tool")
def _fail_tool_advice(
    state: _ToolAdviceState,
    error: BaseException,
) -> None:
    state.invocation.fail(error)


def _wrap_tool_function(
    tool_function: ToolFunction,
    telemetry_handler: TelemetryHandler,
):
    if inspect.iscoroutinefunction(tool_function):

        @functools.wraps(tool_function)
        async def wrapped_function(*args, **kwargs):
            state = _prepare_tool_advice(
                tool_function,
                telemetry_handler,
                args,
                kwargs,
            )
            try:
                result = await tool_function(*args, **kwargs)
            except BaseException as error:
                if state is not None:
                    _fail_tool_advice(state, error)
                raise
            if state is not None:
                _complete_tool_advice(state, result)
            return result
    else:

        @functools.wraps(tool_function)
        def wrapped_function(*args, **kwargs):
            state = _prepare_tool_advice(
                tool_function,
                telemetry_handler,
                args,
                kwargs,
            )
            try:
                result = tool_function(*args, **kwargs)
            except BaseException as error:
                if state is not None:
                    _fail_tool_advice(state, error)
                raise
            if state is not None:
                _complete_tool_advice(state, result)
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
