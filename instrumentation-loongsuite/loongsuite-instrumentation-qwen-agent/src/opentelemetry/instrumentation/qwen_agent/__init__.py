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
Qwen-Agent instrumentation supporting `qwen-agent >= 0.0.20`.

Usage
-----
.. code:: python

    from opentelemetry.instrumentation.qwen_agent import QwenAgentInstrumentor
    from qwen_agent.agents import Assistant

    QwenAgentInstrumentor().instrument()

    bot = Assistant(
        llm={'model': 'qwen-max', 'model_type': 'qwen_dashscope'},
        name='my-assistant',
        system_message='You are a helpful assistant.',
    )

    messages = [{'role': 'user', 'content': 'Hello!'}]
    for responses in bot.run(messages):
        pass

    QwenAgentInstrumentor().uninstrument()

API
---
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.qwen_agent.package import _instruments
from opentelemetry.instrumentation.qwen_agent.patch import (
    wrap_agent_call_llm,
    wrap_agent_call_tool,
    wrap_agent_run,
    wrap_chat_model_chat,
)
from opentelemetry.instrumentation.qwen_agent.version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

_AGENT_MODULE = "qwen_agent.agent"
_LLM_MODULE = "qwen_agent.llm.base"

__all__ = ["QwenAgentInstrumentor", "__version__"]


class QwenAgentInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for Qwen-Agent framework.

    Instruments the following components:
    - Agent.run(): Agent execution spans (invoke_agent)
      (run_nonstream is NOT wrapped separately — it calls run() internally,
       so the invoke_agent span is created once by the run() wrapper.)
    - Agent._call_llm(): ReAct step spans (only for agents with tools)
    - BaseChatModel.chat(): LLM call spans (chat)
    - Agent._call_tool(): Tool execution spans (execute_tool)
    """

    def __init__(self):
        super().__init__()
        self._handler = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Qwen-Agent instrumentation."""
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        self._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        # Instrument Agent.run() - agent execution entry point (generator)
        try:
            wrap_function_wrapper(
                module=_AGENT_MODULE,
                name="Agent.run",
                wrapper=lambda wrapped, instance, args, kwargs: wrap_agent_run(
                    wrapped, instance, args, kwargs, handler=self._handler
                ),
            )
            logger.debug("Instrumented Agent.run")
        except Exception as e:
            logger.warning(f"Failed to instrument Agent.run: {e}")

        # Note: Agent.run_nonstream() is NOT wrapped separately.
        # It internally calls self.run(), which is already instrumented above,
        # so a single invoke_agent span is created per agent execution.

        # Instrument BaseChatModel.chat() - LLM calls
        try:
            wrap_function_wrapper(
                module=_LLM_MODULE,
                name="BaseChatModel.chat",
                wrapper=lambda wrapped, instance, args, kwargs: (
                    wrap_chat_model_chat(
                        wrapped, instance, args, kwargs, handler=self._handler
                    )
                ),
            )
            logger.debug("Instrumented BaseChatModel.chat")
        except Exception as e:
            logger.warning(f"Failed to instrument BaseChatModel.chat: {e}")

        # Instrument Agent._call_llm() - ReAct step tracking (only fires for agents with tools)
        try:
            wrap_function_wrapper(
                module=_AGENT_MODULE,
                name="Agent._call_llm",
                wrapper=lambda wrapped, instance, args, kwargs: (
                    wrap_agent_call_llm(
                        wrapped, instance, args, kwargs, handler=self._handler
                    )
                ),
            )
            logger.debug("Instrumented Agent._call_llm")
        except Exception as e:
            logger.warning(f"Failed to instrument Agent._call_llm: {e}")

        # Instrument Agent._call_tool() - tool execution
        try:
            wrap_function_wrapper(
                module=_AGENT_MODULE,
                name="Agent._call_tool",
                wrapper=lambda wrapped, instance, args, kwargs: (
                    wrap_agent_call_tool(
                        wrapped, instance, args, kwargs, handler=self._handler
                    )
                ),
            )
            logger.debug("Instrumented Agent._call_tool")
        except Exception as e:
            logger.warning(f"Failed to instrument Agent._call_tool: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Qwen-Agent instrumentation."""
        try:
            import qwen_agent.agent  # noqa: PLC0415

            unwrap(qwen_agent.agent.Agent, "run")
            logger.debug("Uninstrumented Agent.run")
        except Exception as e:
            logger.warning(f"Failed to uninstrument Agent.run: {e}")

        try:
            import qwen_agent.llm.base  # noqa: PLC0415

            unwrap(qwen_agent.llm.base.BaseChatModel, "chat")
            logger.debug("Uninstrumented BaseChatModel.chat")
        except Exception as e:
            logger.warning(f"Failed to uninstrument BaseChatModel.chat: {e}")

        try:
            import qwen_agent.agent  # noqa: PLC0415

            unwrap(qwen_agent.agent.Agent, "_call_llm")
            logger.debug("Uninstrumented Agent._call_llm")
        except Exception as e:
            logger.warning(f"Failed to uninstrument Agent._call_llm: {e}")

        try:
            import qwen_agent.agent  # noqa: PLC0415

            unwrap(qwen_agent.agent.Agent, "_call_tool")
            logger.debug("Uninstrumented Agent._call_tool")
        except Exception as e:
            logger.warning(f"Failed to uninstrument Agent._call_tool: {e}")

        self._handler = None
