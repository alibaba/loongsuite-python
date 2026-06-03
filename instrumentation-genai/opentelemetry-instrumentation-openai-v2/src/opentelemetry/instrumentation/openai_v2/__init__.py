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

"""
OpenAI client instrumentation supporting `openai`, it can be enabled by
using ``OpenAIInstrumentor``.

.. _openai: https://pypi.org/project/openai/

Usage
-----

.. code:: python

    from openai import OpenAI
    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

    OpenAIInstrumentor().instrument()

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Write a short poem on open telemetry."},
        ],
    )

API
---
"""

import logging
from importlib import import_module
from typing import Collection

from wrapt import wrap_function_wrapper

from opentelemetry._logs import get_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai_v2.package import _instruments
from opentelemetry.instrumentation.openai_v2.utils import is_content_enabled
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import get_tracer
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
)
from opentelemetry.util.genai.types import ContentCapturingMode
from opentelemetry.util.genai.utils import (
    get_content_capturing_mode,
    is_experimental_mode,
)

from .instruments import Instruments
from .patch import (
    async_chat_completions_create_v_new,
    async_chat_completions_create_v_old,
    async_embeddings_create,
    async_responses_create_v_new,
    async_responses_parse_v_new,
    async_responses_retrieve_v_new,
    chat_completions_create_v_new,
    chat_completions_create_v_old,
    embeddings_create,
    responses_create_v_new,
    responses_parse_v_new,
    responses_retrieve_v_new,
)

_logger = logging.getLogger(__name__)


def _wrap_function_wrapper_if_available(module, name, wrapper):
    try:
        imported_module = import_module(module)
        target = imported_module
        for part in name.split(".")[:-1]:
            target = getattr(target, part)
        getattr(target, name.split(".")[-1])
    except (AttributeError, ModuleNotFoundError) as exc:
        _logger.debug(
            "Skipping optional OpenAI wrapper %s.%s: %s",
            module,
            name,
            exc,
        )
        return
    wrap_function_wrapper(module=module, name=name, wrapper=wrapper)


def _unwrap_if_available(module_name, class_name, method_name):
    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        return
    cls = getattr(module, class_name, None)
    if cls is not None and hasattr(cls, method_name):
        unwrap(cls, method_name)


class OpenAIInstrumentor(BaseInstrumentor):
    def __init__(self):
        self._meter = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        """Enable OpenAI instrumentation."""

        latest_experimental_enabled = is_experimental_mode()
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(
            __name__,
            "",
            tracer_provider,
            schema_url=Schemas.V1_30_0.value,  # only used on the legacy path
        )
        logger_provider = kwargs.get("logger_provider")
        logger = get_logger(
            __name__,
            "",
            logger_provider=logger_provider,
            schema_url=Schemas.V1_30_0.value,  # only used on the legacy path
        )
        meter_provider = kwargs.get("meter_provider")
        self._meter = get_meter(
            __name__,
            "",
            meter_provider,
            schema_url=Schemas.V1_30_0.value,  # only used on the legacy path
        )

        instruments = Instruments(self._meter)

        content_mode = (
            get_content_capturing_mode()
            if latest_experimental_enabled
            else ContentCapturingMode.NO_CONTENT
        )
        handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        wrap_function_wrapper(
            module="openai.resources.chat.completions",
            name="Completions.create",
            wrapper=(
                chat_completions_create_v_new(handler, content_mode)
                if latest_experimental_enabled
                else chat_completions_create_v_old(
                    tracer, logger, instruments, is_content_enabled()
                )
            ),
        )

        wrap_function_wrapper(
            module="openai.resources.chat.completions",
            name="AsyncCompletions.create",
            wrapper=(
                async_chat_completions_create_v_new(handler, content_mode)
                if latest_experimental_enabled
                else async_chat_completions_create_v_old(
                    tracer, logger, instruments, is_content_enabled()
                )
            ),
        )

        # Add instrumentation for the embeddings API
        wrap_function_wrapper(
            module="openai.resources.embeddings",
            name="Embeddings.create",
            wrapper=embeddings_create(
                tracer, instruments, latest_experimental_enabled
            ),
        )

        wrap_function_wrapper(
            module="openai.resources.embeddings",
            name="AsyncEmbeddings.create",
            wrapper=async_embeddings_create(
                tracer, instruments, latest_experimental_enabled
            ),
        )

        if latest_experimental_enabled:
            _wrap_function_wrapper_if_available(
                module="openai.resources.responses",
                name="Responses.create",
                wrapper=responses_create_v_new(handler, content_mode),
            )
            _wrap_function_wrapper_if_available(
                module="openai.resources.responses",
                name="Responses.parse",
                wrapper=responses_parse_v_new(handler, content_mode),
            )
            _wrap_function_wrapper_if_available(
                module="openai.resources.responses",
                name="Responses.retrieve",
                wrapper=responses_retrieve_v_new(handler, content_mode),
            )

            _wrap_function_wrapper_if_available(
                module="openai.resources.responses",
                name="AsyncResponses.create",
                wrapper=async_responses_create_v_new(handler, content_mode),
            )
            _wrap_function_wrapper_if_available(
                module="openai.resources.responses",
                name="AsyncResponses.parse",
                wrapper=async_responses_parse_v_new(handler, content_mode),
            )
            _wrap_function_wrapper_if_available(
                module="openai.resources.responses",
                name="AsyncResponses.retrieve",
                wrapper=async_responses_retrieve_v_new(handler, content_mode),
            )

    def _uninstrument(self, **kwargs):
        import openai  # pylint: disable=import-outside-toplevel  # noqa: PLC0415

        unwrap(openai.resources.chat.completions.Completions, "create")
        unwrap(openai.resources.chat.completions.AsyncCompletions, "create")
        unwrap(openai.resources.embeddings.Embeddings, "create")
        unwrap(openai.resources.embeddings.AsyncEmbeddings, "create")
        _unwrap_if_available(
            "openai.resources.responses", "Responses", "create"
        )
        _unwrap_if_available(
            "openai.resources.responses", "Responses", "parse"
        )
        _unwrap_if_available(
            "openai.resources.responses", "Responses", "retrieve"
        )
        _unwrap_if_available(
            "openai.resources.responses", "AsyncResponses", "create"
        )
        _unwrap_if_available(
            "openai.resources.responses", "AsyncResponses", "parse"
        )
        _unwrap_if_available(
            "openai.resources.responses", "AsyncResponses", "retrieve"
        )
