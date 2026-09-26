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
OpenTelemetry Anthropic Instrumentation
========================================

Instrumentation for the Anthropic Python SDK.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
    import anthropic

    # Enable instrumentation
    AnthropicInstrumentor().instrument()

    # Use Anthropic client normally
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}]
    )

Configuration
-------------

Message content capture can be enabled by setting the environment variable:
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true``

Or via the ``OTEL_INSTRUMENTATION_GENAI_EXPERIMENTAL_CONTENT_CAPTURING_MODE``
environment variable (values: ``none``, ``span``, ``event``, ``all``).

API
---
"""

from typing import Any, Collection

from wrapt import (
    wrap_function_wrapper,  # pyright: ignore[reportUnknownVariableType]
)

from opentelemetry.instrumentation.anthropic.package import _instruments
from opentelemetry.instrumentation.anthropic.patch import (
    async_messages_create,
    messages_create,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import ContentCapturingMode
from opentelemetry.util.genai.utils import (
    get_content_capturing_mode,
    is_experimental_mode,
)


class AnthropicInstrumentor(BaseInstrumentor):
    """An instrumentor for the Anthropic Python SDK.

    This instrumentor will automatically trace Anthropic API calls and
    optionally capture message content as events.

    Supported features:
    - Sync and async Messages.create
    - Streaming responses (stream=True)
    - Content capture (input/output messages, system instructions)
    - Tool calling (tool definitions, tool use blocks, tool results)
    - Token usage metrics (including cache tokens)
    - Error handling and exception recording
    """

    def __init__(self) -> None:
        super().__init__()

    # pylint: disable=no-self-use
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Anthropic instrumentation.

        Args:
            **kwargs: Optional arguments
                - tracer_provider: TracerProvider instance
                - meter_provider: MeterProvider instance
                - logger_provider: LoggerProvider instance
        """
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        content_mode = (
            get_content_capturing_mode()
            if is_experimental_mode()
            else ContentCapturingMode.NO_CONTENT
        )

        # Patch sync Messages.create
        wrap_function_wrapper(
            "anthropic.resources.messages",
            "Messages.create",
            messages_create(handler, content_mode),
        )

        # Patch async AsyncMessages.create
        wrap_function_wrapper(
            "anthropic.resources.messages",
            "AsyncMessages.create",
            async_messages_create(handler, content_mode),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Anthropic instrumentation.

        This removes all patches applied during instrumentation.
        """
        import anthropic  # pylint: disable=import-outside-toplevel  # noqa: PLC0415

        unwrap(
            anthropic.resources.messages.Messages,  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]
            "create",
        )
        unwrap(
            anthropic.resources.messages.AsyncMessages,  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]
            "create",
        )
