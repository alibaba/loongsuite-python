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

import importlib
import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.agno._wrapper import (
    AgnoAgentWrapper,
    AgnoFunctionCallWrapper,
    AgnoModelWrapper,
)
from opentelemetry.instrumentation.agno.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

"""OpenTelemetry exporters for Agno https://github.com/agno-agi/agno"""

_AGENT = "agno.agent"
_MODULE = "agno.models.base"
_TOOLKIT = "agno.tools.function"
_logger = logging.getLogger(__name__)
__all__ = ["AgnoInstrumentor"]

_INNER_MODEL_WRAPPERS = (
    ("Model._process_model_response", "process_model_response"),
    ("Model._aprocess_model_response", "aprocess_model_response"),
    ("Model.process_response_stream", "process_response_stream"),
    ("Model.aprocess_response_stream", "aprocess_response_stream"),
    ("Model.run_function_calls", "run_function_calls"),
    ("Model.arun_function_calls", "arun_function_calls"),
)

_FALLBACK_MODEL_WRAPPERS = (
    ("Model.response", "response"),
    ("Model.aresponse", "aresponse"),
    ("Model.response_stream", "response_stream"),
    ("Model.aresponse_stream", "aresponse_stream"),
)


def _has_wrap_target(module_name: str, name: str) -> bool:
    try:
        target = importlib.import_module(module_name)
        for part in name.split("."):
            target = getattr(target, part)
    except Exception:
        return False
    return True


def _wrap_model_methods(model_wrapper: AgnoModelWrapper) -> None:
    if all(
        _has_wrap_target(_MODULE, name)
        for name, _wrapper_name in _INNER_MODEL_WRAPPERS
    ):
        for name, wrapper_name in _INNER_MODEL_WRAPPERS:
            wrap_function_wrapper(
                module=_MODULE,
                name=name,
                wrapper=getattr(model_wrapper, wrapper_name),
            )
        return

    _logger.warning(
        "Agno inner model hooks are unavailable; falling back to outer "
        "Model.response wrappers."
    )
    for name, wrapper_name in _FALLBACK_MODEL_WRAPPERS:
        wrap_function_wrapper(
            module=_MODULE,
            name=name,
            wrapper=getattr(model_wrapper, wrapper_name),
        )


def _unwrap_if_present(target: Any, name: str) -> None:
    if hasattr(target, name):
        unwrap(target, name)


class AgnoInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for agno.
    """

    def __init__(self):
        super().__init__()
        self._handler = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        try:
            from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
                get_extended_telemetry_handler,
            )
        except ImportError as exc:
            raise RuntimeError(
                "loongsuite-instrumentation-agno requires "
                "opentelemetry-util-genai with ExtendedTelemetryHandler support"
            ) from exc

        tracer_provider = kwargs.get("tracer_provider")
        logger_provider = kwargs.get("logger_provider")
        self._handler = get_extended_telemetry_handler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
        )

        agent_wrapper = AgnoAgentWrapper(self._handler)
        function_call_wrapper = AgnoFunctionCallWrapper(self._handler)
        model_wrapper = AgnoModelWrapper(self._handler)

        # Wrap the agent run
        wrap_function_wrapper(
            module=_AGENT,
            name="Agent.run",
            wrapper=agent_wrapper.run,
        )
        wrap_function_wrapper(
            module=_AGENT,
            name="Agent.arun",
            wrapper=agent_wrapper.arun,
        )

        # Wrap the function
        wrap_function_wrapper(
            module=_TOOLKIT,
            name="FunctionCall.execute",
            wrapper=function_call_wrapper.execute,
        )
        wrap_function_wrapper(
            module=_TOOLKIT,
            name="FunctionCall.aexecute",
            wrapper=function_call_wrapper.aexecute,
        )

        # Wrap the model. Prefer Agno's per-request internals so a tool-call
        # loop emits one LLM span per provider call.
        _wrap_model_methods(model_wrapper)

    def _uninstrument(self, **kwargs: Any) -> None:
        # Unwrap the agent call function
        import agno.agent  # noqa: PLC0415

        unwrap(agno.agent.Agent, "run")
        unwrap(agno.agent.Agent, "arun")

        # Unwrap the function call
        import agno.tools.function  # noqa: PLC0415

        unwrap(agno.tools.function.FunctionCall, "execute")
        unwrap(agno.tools.function.FunctionCall, "aexecute")

        # Unwrap the model
        import agno.models.base  # noqa: PLC0415

        for name, _wrapper_name in (
            *_INNER_MODEL_WRAPPERS,
            *_FALLBACK_MODEL_WRAPPERS,
        ):
            _unwrap_if_present(agno.models.base.Model, name.split(".", 1)[1])
        self._handler = None
