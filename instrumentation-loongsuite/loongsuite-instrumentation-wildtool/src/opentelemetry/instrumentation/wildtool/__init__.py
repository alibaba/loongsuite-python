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

"""OpenTelemetry WildToolBench Instrumentation"""

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.wildtool._wrappers import (
    WildToolAgentWrapper,
    WildToolChainWrapper,
    WildToolEntryWrapper,
    WildToolParseWrapper,
    WildToolRequestWrapper,
)
from opentelemetry.instrumentation.wildtool.package import _instruments
from opentelemetry.instrumentation.wildtool.version import __version__
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

_LLM_RESPONSE_GEN_MODULE = "wtb._llm_response_generation"
_BASE_HANDLER_MODULE = "wtb.model_handler.base_handler"

__all__ = ["WildToolInstrumentor", "__version__"]


class WildToolInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for WildToolBench framework."""

    def __init__(self):
        super().__init__()
        self._handler = None
        # Track concrete handler subclasses whose abstract _request_tool_call /
        # _parse_api_response we have already wrapped, so we can unwrap on
        # uninstrument and avoid double-wrapping.
        self._patched_handler_classes: set = set()
        self._request_wrapper = None
        self._parse_wrapper = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        self._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        self._request_wrapper = WildToolRequestWrapper(self._handler)
        self._parse_wrapper = WildToolParseWrapper(self._handler)

        # P1: ENTRY span
        try:
            wrap_function_wrapper(
                _LLM_RESPONSE_GEN_MODULE,
                "multi_threaded_inference",
                WildToolEntryWrapper(self._handler),
            )
        except Exception as e:
            logger.warning(
                "Failed to instrument multi_threaded_inference: %s", e
            )

        # P2: AGENT span
        try:
            wrap_function_wrapper(
                _BASE_HANDLER_MODULE,
                "BaseHandler.inference_multi_turn",
                WildToolAgentWrapper(self._handler),
            )
        except Exception as e:
            logger.warning("Failed to instrument inference_multi_turn: %s", e)

        # P3: CHAIN span (+ STEP + TOOL management).
        # The chain wrapper also lazily patches the concrete subclass'
        # `_request_tool_call` / `_parse_api_response` on first use, so that
        # subclasses overriding the abstract base methods are still
        # intercepted (P4 / P5).
        try:
            wrap_function_wrapper(
                _BASE_HANDLER_MODULE,
                "BaseHandler.inference_and_eval_multi_step",
                WildToolChainWrapper(self._handler, self),
            )
        except Exception as e:
            logger.warning(
                "Failed to instrument inference_and_eval_multi_step: %s", e
            )

    def ensure_handler_class_patched(self, handler_cls) -> None:
        """Lazily wrap the concrete handler subclass' P4/P5 methods.

        WildToolBench declares ``_request_tool_call`` and ``_parse_api_response``
        as abstract on ``BaseHandler``, but real handlers (and tests) override
        them. Python method resolution dispatches directly to the override and
        therefore never reaches a wrapper installed on the base class. We
        instead wrap the override on first invocation per subclass.
        """
        if handler_cls in self._patched_handler_classes:
            return
        self._patched_handler_classes.add(handler_cls)

        module_name = handler_cls.__module__
        cls_name = handler_cls.__name__
        for method, wrapper in (
            ("_request_tool_call", self._request_wrapper),
            ("_parse_api_response", self._parse_wrapper),
        ):
            if method not in handler_cls.__dict__:
                continue
            try:
                wrap_function_wrapper(
                    module_name,
                    f"{cls_name}.{method}",
                    wrapper,
                )
            except Exception as e:
                logger.debug(
                    "Failed to wrap %s.%s.%s: %s",
                    module_name,
                    cls_name,
                    method,
                    e,
                )

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            import wtb._llm_response_generation as llm_gen

            unwrap(llm_gen, "multi_threaded_inference")
        except Exception as e:
            logger.debug(
                "Failed to uninstrument multi_threaded_inference: %s", e
            )

        try:
            import wtb.model_handler.base_handler as bh

            unwrap(bh.BaseHandler, "inference_multi_turn")
            unwrap(bh.BaseHandler, "inference_and_eval_multi_step")
        except Exception as e:
            logger.debug("Failed to uninstrument BaseHandler methods: %s", e)

        for cls in list(self._patched_handler_classes):
            for method in ("_request_tool_call", "_parse_api_response"):
                if method in cls.__dict__:
                    try:
                        unwrap(cls, method)
                    except Exception as e:
                        logger.debug(
                            "Failed to unwrap %s.%s: %s",
                            cls.__name__,
                            method,
                            e,
                        )
        self._patched_handler_classes.clear()
        self._request_wrapper = None
        self._parse_wrapper = None
        self._handler = None
