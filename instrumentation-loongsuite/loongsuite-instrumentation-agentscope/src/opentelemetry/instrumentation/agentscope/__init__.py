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
AgentScope instrumentation supporting `agentscope >= 1.0.0, < 3.0.0`.

Usage
-----
.. code:: python
    import asyncio
    from opentelemetry.instrumentation.agentscope import AgentScopeInstrumentor
    from agentscope.model import DashScopeChatModel
    import agentscope

    AgentScopeInstrumentor().instrument()

    model = DashScopeChatModel(model_name="qwen-max")

    messages = [{"role": "user", "content": "Hello, how are you?"}]

    async def call_model():
        response = await model(messages)
        if hasattr(response, "__aiter__"):
            result = []
            async for chunk in response:
                result.append(chunk)
            return result[-1] if result else response
        return response

    result = asyncio.run(call_model())

    AgentScopeInstrumentor().uninstrument()

API
---
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.agentscope.package import (
    get_installed_instrumentation_dependencies,
)
from opentelemetry.instrumentation.agentscope.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

_MODEL_MODULE = "agentscope.model"
_AGENT_MODULE = "agentscope.agent"
_EMBEDDING_MODULE = "agentscope.embedding"
_TOOL_MODULE = "agentscope.tool"
_FORMATTER_MODULE = "agentscope.formatter"

__all__ = ["AgentScopeInstrumentor"]


class AgentScopeInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for AgentScope framework.

    Agent ``__call__`` instrumentation keeps per-call state on the instance;
    do not run overlapping ``await agent(...)`` on the same agent without
    external synchronization (aligned with AgentScope's single in-flight
    ``_reply_task`` design).
    """

    def __init__(self):
        super().__init__()
        self._tracer = None  # Only used for Formatter instrumentation
        self._handler = (
            None  # ExtendedTelemetryHandler handles all other operations
        )
        self._agentscope_major = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return get_installed_instrumentation_dependencies()

    def _setup_tracing_patch(self, wrapped, instance, args, kwargs):
        """Replace setup_tracing with no-op to use OTEL instead."""
        pass

    def _check_tracing_enabled_patch(self, wrapped, instance, args, kwargs):
        """Return False to disable tracing in native AgentScope library."""
        return False

    def _instrument(self, **kwargs: Any) -> None:
        """Enable AgentScope instrumentation."""
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        # ExtendedTelemetryHandler internally creates tracer/meter/logger with correct schema (V1_37_0)
        self._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        # Create a separate tracer for Formatter instrumentation (it doesn't use handler)
        self._tracer = trace_api.get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=Schemas.V1_37_0.value,
        )

        self._agentscope_major = _get_agentscope_major()
        if self._agentscope_major >= 2:
            self._instrument_v2()
        else:
            self._instrument_v1()

    def _instrument_v1(self) -> None:
        from ._wrapper import (  # noqa: PLC0415
            AgentScopeAgentWrapper,
            AgentScopeChatModelWrapper,
            AgentScopeEmbeddingModelWrapper,
        )
        from .patch import (  # noqa: PLC0415
            wrap_formatter_format,
            wrap_tool_call,
        )

        # Instrument ChatModelBase
        try:
            chat_wrapper = AgentScopeChatModelWrapper(handler=self._handler)
            wrap_function_wrapper(
                module=_MODEL_MODULE,
                name="ChatModelBase.__init__",
                wrapper=chat_wrapper,
            )
            logger.debug("Instrumented ChatModelBase")
        except Exception as e:
            logger.warning(f"Failed to instrument ChatModelBase: {e}")

        # Instrument AgentBase
        try:
            agent_wrapper = AgentScopeAgentWrapper(handler=self._handler)
            wrap_function_wrapper(
                module=_AGENT_MODULE,
                name="AgentBase.__init__",
                wrapper=agent_wrapper,
            )
            logger.debug("Instrumented AgentBase")
        except Exception as e:
            logger.warning(f"Failed to instrument AgentBase: {e}")

        # Instrument EmbeddingModelBase
        try:
            embedding_wrapper = AgentScopeEmbeddingModelWrapper(
                handler=self._handler
            )
            wrap_function_wrapper(
                module=_EMBEDDING_MODULE,
                name="EmbeddingModelBase.__init__",
                wrapper=embedding_wrapper,
            )
            logger.debug("Instrumented EmbeddingModelBase")
        except Exception as e:
            logger.warning(f"Failed to instrument EmbeddingModelBase: {e}")

        # Instrument Toolkit
        try:

            def wrap_tool_with_handler(wrapped, instance, args, kwargs):
                return wrap_tool_call(
                    wrapped, instance, args, kwargs, handler=self._handler
                )

            wrap_function_wrapper(
                module=_TOOL_MODULE,
                name="Toolkit.call_tool_function",
                wrapper=wrap_tool_with_handler,
            )
            logger.debug("Instrumented Toolkit")
        except Exception as e:
            logger.warning(f"Failed to instrument Toolkit: {e}")

        # Instrument Formatter
        try:

            def wrap_formatter_with_tracer(wrapped, instance, args, kwargs):
                return wrap_formatter_format(
                    wrapped, instance, args, kwargs, tracer=self._tracer
                )

            wrap_function_wrapper(
                module=_FORMATTER_MODULE,
                name="TruncatedFormatterBase.format",
                wrapper=wrap_formatter_with_tracer,
            )
            logger.debug("Instrumented TruncatedFormatterBase")
        except Exception as e:
            logger.warning(f"Failed to instrument TruncatedFormatterBase: {e}")

        # Patch setup_tracing to be a no-op
        try:
            wrap_function_wrapper(
                module="agentscope.tracing",
                name="setup_tracing",
                wrapper=self._setup_tracing_patch,
            )
            logger.debug("Patched setup_tracing")
        except Exception as e:
            logger.warning(f"Failed to patch setup_tracing: {e}")

        # Patch _check_tracing_enabled to return False
        # We always want to disable tracing in native AgentScope library
        try:
            wrap_function_wrapper(
                module="agentscope.tracing._trace",
                name="_check_tracing_enabled",
                wrapper=self._check_tracing_enabled_patch,
            )
            logger.debug("Patched _check_tracing_enabled")
        except Exception as e:
            logger.warning(f"Failed to patch _check_tracing_enabled: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable AgentScope instrumentation."""
        del kwargs
        if self._agentscope_major is None:
            self._agentscope_major = _get_agentscope_major()
        if self._agentscope_major >= 2:
            self._uninstrument_v2()
        else:
            self._uninstrument_v1()
        self._handler = None
        self._tracer = None
        self._agentscope_major = None

    def _uninstrument_v1(self) -> None:
        try:
            from ._wrapper import (  # noqa: PLC0415
                AgentScopeAgentWrapper,
                AgentScopeChatModelWrapper,
                AgentScopeEmbeddingModelWrapper,
            )
        except Exception as e:
            logger.warning(f"Failed to import AgentScope wrappers: {e}")
            AgentScopeAgentWrapper = None
            AgentScopeChatModelWrapper = None
            AgentScopeEmbeddingModelWrapper = None

        try:
            if AgentScopeChatModelWrapper is not None:
                AgentScopeChatModelWrapper.restore_original_methods()
                logger.debug("Restored ChatModelBase methods")
        except Exception as e:
            logger.warning(f"Failed to restore ChatModelBase: {e}")

        try:
            if AgentScopeAgentWrapper is not None:
                AgentScopeAgentWrapper.restore_original_methods()
                logger.debug("Restored AgentBase methods")
        except Exception as e:
            logger.warning(f"Failed to restore AgentBase: {e}")

        try:
            if AgentScopeEmbeddingModelWrapper is not None:
                AgentScopeEmbeddingModelWrapper.restore_original_methods()
                logger.debug("Restored EmbeddingModelBase methods")
        except Exception as e:
            logger.warning(f"Failed to restore EmbeddingModelBase: {e}")

        try:
            import agentscope.model  # noqa: PLC0415

            unwrap(agentscope.model.ChatModelBase, "__init__")
            logger.debug("Uninstrumented ChatModelBase")
        except Exception as e:
            logger.warning(f"Failed to uninstrument ChatModelBase: {e}")

        try:
            import agentscope.agent  # noqa: PLC0415

            unwrap(agentscope.agent.AgentBase, "__init__")
            logger.debug("Uninstrumented AgentBase")
        except Exception as e:
            logger.warning(f"Failed to uninstrument AgentBase: {e}")

        try:
            import agentscope.embedding  # noqa: PLC0415

            unwrap(agentscope.embedding.EmbeddingModelBase, "__init__")
            logger.debug("Uninstrumented EmbeddingModelBase")
        except Exception as e:
            logger.warning(f"Failed to uninstrument EmbeddingModelBase: {e}")

        try:
            import agentscope.tool  # noqa: PLC0415

            unwrap(agentscope.tool.Toolkit, "call_tool_function")
            logger.debug("Uninstrumented Toolkit")
        except Exception as e:
            logger.warning(f"Failed to uninstrument Toolkit: {e}")

        try:
            import agentscope.formatter  # noqa: PLC0415

            unwrap(agentscope.formatter.TruncatedFormatterBase, "format")
            logger.debug("Uninstrumented TruncatedFormatterBase")
        except Exception as e:
            logger.warning(
                f"Failed to uninstrument TruncatedFormatterBase: {e}"
            )

        try:
            import agentscope.tracing  # noqa: PLC0415

            unwrap(agentscope.tracing, "setup_tracing")
            logger.debug("Uninstrumented setup_tracing")
        except Exception as e:
            logger.warning(f"Failed to uninstrument setup_tracing: {e}")

        try:
            import agentscope.tracing._trace as agentscope_tracing_trace  # noqa: PLC0415

            unwrap(agentscope_tracing_trace, "_check_tracing_enabled")
            logger.debug("Uninstrumented _check_tracing_enabled")
        except Exception as e:
            logger.warning(
                f"Failed to uninstrument _check_tracing_enabled: {e}"
            )

    def _instrument_v2(self) -> None:
        from ._v2_middleware import (  # noqa: PLC0415
            AgentScopeV2Middleware,
            append_loongsuite_middleware,
        )

        try:

            def wrap_agent_init(wrapped, instance, args, kwargs):
                args, kwargs = append_loongsuite_middleware(
                    args,
                    kwargs,
                    AgentScopeV2Middleware(handler=lambda: self._handler),
                )
                return wrapped(*args, **kwargs)

            wrap_function_wrapper(
                module=_AGENT_MODULE,
                name="Agent.__init__",
                wrapper=wrap_agent_init,
            )
            logger.debug("Instrumented AgentScope v2 Agent")
        except Exception as e:
            logger.warning(f"Failed to instrument AgentScope v2 Agent: {e}")

    def _uninstrument_v2(self) -> None:
        try:
            import agentscope.agent  # noqa: PLC0415

            unwrap(agentscope.agent.Agent, "__init__")
            logger.debug("Uninstrumented AgentScope v2 Agent")
        except Exception as e:
            logger.warning(f"Failed to uninstrument AgentScope v2 Agent: {e}")


def _get_agentscope_major() -> int:
    try:
        installed_version = metadata_version("agentscope")
    except PackageNotFoundError:
        return 1

    major_text = installed_version.split(".", 1)[0]
    try:
        return int(major_text)
    except ValueError:
        return 1
