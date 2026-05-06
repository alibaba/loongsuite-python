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

"""LoongSuite BFCL v4 (Berkeley Function Call Leaderboard) instrumentation.

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.bfclv4 import BFCLv4Instrumentor

    BFCLv4Instrumentor().instrument()
    # ... run BFCL ...
    BFCLv4Instrumentor().uninstrument()

API
---
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Collection, List, Tuple

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.bfclv4.internal.wrappers import (
    BaseHandlerInferenceWrapper,
    ExecuteFuncCallWrapper,
    GenerateResultsWrapper,
    QueryWrapper,
    TurnBumpWrapper,
)
from opentelemetry.instrumentation.bfclv4.package import _instruments
from opentelemetry.instrumentation.bfclv4.utils import GenAIHookHelper
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

logger = logging.getLogger(__name__)

__all__ = ["BFCLv4Instrumentor"]


_GENERATE_RESULTS_MODULE = "bfcl_eval._llm_response_generation"
_GENERATE_RESULTS_NAME = "generate_results"

_BASE_HANDLER_MODULE = "bfcl_eval.model_handler.base_handler"
_BASE_HANDLER_NAME = "BaseHandler.inference"

_EXECUTE_TOOL_MODULE = (
    "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils"
)
_EXECUTE_TOOL_NAME = "execute_multi_turn_func_call"


# ``MODEL_CONFIG_MAPPING`` already imports every concrete handler at module
# load time, so iterating over its values gives us the canonical handler
# class set without risking new vendor SDK imports.
def _iter_handler_classes() -> List[type]:
    try:
        from bfcl_eval.constants.model_config import (  # noqa: PLC0415
            MODEL_CONFIG_MAPPING,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "bfclv4: cannot import MODEL_CONFIG_MAPPING: %s", exc
        )
        return []

    classes: List[type] = []
    seen_class_ids: set[int] = set()
    for cfg in MODEL_CONFIG_MAPPING.values():
        cls = getattr(cfg, "model_handler", None)
        if cls is None or not isinstance(cls, type):
            continue
        if id(cls) in seen_class_ids:
            continue
        seen_class_ids.add(id(cls))
        classes.append(cls)
    return classes


class BFCLv4Instrumentor(BaseInstrumentor):
    """An instrumentor for the BFCL v4 (``bfcl_eval``) framework."""

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(self, "_wrapped_query_methods"):
            self._wrapped_query_methods: List[Tuple[type, str]] = []
        if not hasattr(self, "_wrapped_turn_methods"):
            self._wrapped_turn_methods: List[Tuple[type, str]] = []
        if not hasattr(self, "_entry_wrapped"):
            self._entry_wrapped = False
        if not hasattr(self, "_inference_wrapped"):
            self._inference_wrapped = False
        if not hasattr(self, "_tool_wrapped"):
            self._tool_wrapped = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    # ------------------------------------------------------------------
    # _instrument

    def _instrument(self, **kwargs: Any) -> None:  # noqa: D401
        helper = GenAIHookHelper()

        # 1) ENTRY -----------------------------------------------------
        try:
            wrap_function_wrapper(
                module=_GENERATE_RESULTS_MODULE,
                name=_GENERATE_RESULTS_NAME,
                wrapper=GenerateResultsWrapper(helper),
            )
            self._entry_wrapped = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "bfclv4: failed to wrap %s.%s: %s",
                _GENERATE_RESULTS_MODULE,
                _GENERATE_RESULTS_NAME,
                exc,
            )

        # 2) AGENT -----------------------------------------------------
        try:
            wrap_function_wrapper(
                module=_BASE_HANDLER_MODULE,
                name=_BASE_HANDLER_NAME,
                wrapper=BaseHandlerInferenceWrapper(helper),
            )
            self._inference_wrapped = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "bfclv4: failed to wrap %s.%s: %s",
                _BASE_HANDLER_MODULE,
                _BASE_HANDLER_NAME,
                exc,
            )

        # 3) STEP + 4) turn maintenance --------------------------------
        self._instrument_handlers(helper)

        # 5) TOOL ------------------------------------------------------
        try:
            wrap_function_wrapper(
                module=_EXECUTE_TOOL_MODULE,
                name=_EXECUTE_TOOL_NAME,
                wrapper=ExecuteFuncCallWrapper(helper),
            )
            self._tool_wrapped = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "bfclv4: failed to wrap %s.%s: %s",
                _EXECUTE_TOOL_MODULE,
                _EXECUTE_TOOL_NAME,
                exc,
            )

    def _instrument_handlers(self, helper: GenAIHookHelper) -> None:
        # Reflectively wrap every concrete ``_query_FC`` / ``_query_prompting``
        # plus the turn-maintenance helpers; we de-duplicate by function id so
        # subclasses that share an inherited implementation are wrapped only
        # once.
        seen_func_ids: set[int] = set()

        query_pairs = (
            ("_query_FC", "FC"),
            ("_query_prompting", "prompting"),
        )
        turn_pairs = (
            ("add_first_turn_message_FC", True),
            ("add_first_turn_message_prompting", True),
            ("_add_next_turn_user_message_FC", False),
            ("_add_next_turn_user_message_prompting", False),
        )

        for cls in _iter_handler_classes():
            class_dict = getattr(cls, "__dict__", {})
            for method_name, mode in query_pairs:
                method = class_dict.get(method_name)
                if method is None or not callable(method):
                    continue
                key = id(method)
                if key in seen_func_ids:
                    continue
                seen_func_ids.add(key)
                try:
                    wrap_function_wrapper(
                        module=cls.__module__,
                        name=f"{cls.__name__}.{method_name}",
                        wrapper=QueryWrapper(helper, mode),
                    )
                    self._wrapped_query_methods.append((cls, method_name))
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "bfclv4: failed to wrap %s.%s.%s: %s",
                        cls.__module__,
                        cls.__name__,
                        method_name,
                        exc,
                    )

            for method_name, is_first in turn_pairs:
                method = class_dict.get(method_name)
                if method is None or not callable(method):
                    continue
                key = id(method)
                if key in seen_func_ids:
                    continue
                seen_func_ids.add(key)
                try:
                    wrap_function_wrapper(
                        module=cls.__module__,
                        name=f"{cls.__name__}.{method_name}",
                        wrapper=TurnBumpWrapper(reset=is_first),
                    )
                    self._wrapped_turn_methods.append((cls, method_name))
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "bfclv4: failed to wrap %s.%s.%s: %s",
                        cls.__module__,
                        cls.__name__,
                        method_name,
                        exc,
                    )

    # ------------------------------------------------------------------
    # _uninstrument

    def _uninstrument(self, **kwargs: Any) -> None:  # noqa: D401
        if self._tool_wrapped:
            try:
                module = importlib.import_module(_EXECUTE_TOOL_MODULE)
                unwrap(module, _EXECUTE_TOOL_NAME)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "bfclv4: failed to unwrap execute_multi_turn_func_call: %s",
                    exc,
                )
            self._tool_wrapped = False

        for cls, method_name in self._wrapped_query_methods:
            try:
                unwrap(cls, method_name)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "bfclv4: failed to unwrap %s.%s: %s",
                    cls.__name__,
                    method_name,
                    exc,
                )
        self._wrapped_query_methods = []

        for cls, method_name in self._wrapped_turn_methods:
            try:
                unwrap(cls, method_name)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "bfclv4: failed to unwrap %s.%s: %s",
                    cls.__name__,
                    method_name,
                    exc,
                )
        self._wrapped_turn_methods = []

        if self._inference_wrapped:
            try:
                base_module = importlib.import_module(_BASE_HANDLER_MODULE)
                unwrap(base_module.BaseHandler, "inference")
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "bfclv4: failed to unwrap BaseHandler.inference: %s", exc
                )
            self._inference_wrapped = False

        if self._entry_wrapped:
            try:
                module = importlib.import_module(_GENERATE_RESULTS_MODULE)
                unwrap(module, _GENERATE_RESULTS_NAME)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "bfclv4: failed to unwrap generate_results: %s", exc
                )
            self._entry_wrapped = False
