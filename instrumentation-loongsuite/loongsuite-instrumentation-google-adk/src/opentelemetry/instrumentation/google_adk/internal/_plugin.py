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
OpenTelemetry ADK Observability Plugin.

This module implements the core observability plugin using Google ADK's
plugin mechanism with OpenTelemetry GenAI semantic conventions.

This implementation uses ExtendedTelemetryHandler from opentelemetry-util-genai
for standard span and metrics management.
"""

import logging
import timeit
from contextvars import ContextVar, Token
from typing import Any, Dict, List, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    ExecuteToolInvocation,
    InvokeAgentInvocation,
)
from opentelemetry.util.genai.handler import _safe_detach
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)

from ._extractors import AdkAttributeExtractors

_logger = logging.getLogger(__name__)
_ACTIVE_LLM_REQUEST_KEY: ContextVar[Optional[str]] = ContextVar(
    "google_adk_active_llm_request_key", default=None
)


class GoogleAdkObservabilityPlugin(BasePlugin):
    """
    OpenTelemetry ADK Observability Plugin.

    Implements comprehensive observability for Google ADK applications
    following OpenTelemetry GenAI semantic conventions.

    Uses ExtendedTelemetryHandler for standard span lifecycle management
    and automatic metrics recording.
    """

    def __init__(self, handler: ExtendedTelemetryHandler):
        """
        Initialize the observability plugin.

        Args:
            handler: ExtendedTelemetryHandler instance for span/metrics management
        """
        super().__init__(name="opentelemetry_adk_observability")
        self._handler = handler
        self._extractors = AdkAttributeExtractors()

        # Track active invocations for proper callback matching
        self._active_runner_invocations: Dict[str, InvokeAgentInvocation] = {}
        self._active_agent_invocations: Dict[str, InvokeAgentInvocation] = {}
        self._active_llm_invocations: Dict[str, LLMInvocation] = {}
        self._active_tool_invocations: Dict[str, ExecuteToolInvocation] = {}

        # Track user messages and final responses for Runner spans
        self._runner_inputs: Dict[str, types.Content] = {}
        self._runner_outputs: Dict[str, str] = {}

        # Track llm_request -> model mapping to avoid fallback model names
        self._llm_req_models: Dict[str, str] = {}
        self._llm_stream_outputs: Dict[str, str] = {}
        self._llm_context_tokens: Dict[str, Token] = {}

    # ===== Runner Level Callbacks - Top-level invoke_agent span =====

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[Any]:
        """
        Start Runner execution - create top-level invoke_agent span.

        According to OTel GenAI conventions, Runner is treated as a top-level agent.
        """
        try:
            # Extract conversation_id
            conversation_id = None

            conversation_id = self._session_id_from_invocation_context(
                invocation_context
            )

            # Create invocation object
            invocation = InvokeAgentInvocation(
                provider="google_adk",
                agent_name=invocation_context.app_name,
            )

            # Set conversation_id if available
            if conversation_id:
                invocation.conversation_id = conversation_id

            # Set custom attributes
            if hasattr(invocation_context, "app_name"):
                invocation.attributes["google_adk.runner.app_name"] = (
                    invocation_context.app_name
                )

            if hasattr(invocation_context, "invocation_id"):
                invocation.attributes["google_adk.runner.invocation_id"] = (
                    invocation_context.invocation_id
                )

            # Check if we already have a stored user message
            runner_key = self._runner_key(invocation_context)
            if runner_key in self._runner_inputs:
                user_message = self._runner_inputs[runner_key]
                input_messages = self._convert_user_message_to_input_messages(
                    user_message
                )
                invocation.input_messages = input_messages

            # Start invocation (creates span)
            self._handler.start_invoke_agent(invocation)

            # Store invocation for later use
            self._active_runner_invocations[runner_key] = invocation

            _logger.debug(
                f"Started Runner invocation: invoke_agent {invocation_context.app_name}"
            )

        except Exception as e:
            _logger.exception(f"Error in before_run_callback: {e}")

        return None

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        """
        Capture user input for Runner span.

        This callback is triggered when a user message is received.
        """
        try:
            # Store user message for later use in Runner span
            runner_key = self._runner_key(invocation_context)
            self._runner_inputs[runner_key] = user_message

            # Update active Runner invocation if it exists
            invocation = self._active_runner_invocations.get(runner_key)
            if invocation:
                input_messages = self._convert_user_message_to_input_messages(
                    user_message
                )
                invocation.input_messages = input_messages

            _logger.debug(
                f"Captured user message for Runner: {invocation_context.invocation_id}"
            )

        except Exception as e:
            _logger.exception(f"Error in on_user_message_callback: {e}")

        return None  # Don't modify the user message

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        """
        Capture output events for Runner span.

        This callback is triggered for each event generated during execution.
        """
        try:
            # Extract text content from event if available
            event_content = ""
            if hasattr(event, "content") and event.content:
                event_content = self._extract_text_from_content(event.content)
            elif hasattr(event, "data") and event.data:
                event_content = self._extract_text_from_content(event.data)

            if event_content:
                runner_key = self._runner_key(invocation_context)

                # Accumulate output content
                if runner_key not in self._runner_outputs:
                    self._runner_outputs[runner_key] = ""
                self._runner_outputs[runner_key] += event_content

                # Update active Runner invocation
                invocation = self._active_runner_invocations.get(runner_key)
                if invocation:
                    output_messages = [
                        OutputMessage(
                            role="assistant",
                            parts=[
                                Text(content=self._runner_outputs[runner_key])
                            ],
                            finish_reason="stop",
                        )
                    ]
                    invocation.output_messages = output_messages

            _logger.debug(
                f"Captured event for Runner: {invocation_context.invocation_id}"
            )

            if self._is_root_final_event(event, invocation_context):
                self._finish_runner_invocation(invocation_context)

        except Exception as e:
            _logger.exception(f"Error in on_event_callback: {e}")

        return None  # Don't modify the event

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[None]:
        """
        End Runner execution - finish top-level invoke_agent span.
        """
        try:
            self._finish_runner_invocation(invocation_context)

        except Exception as e:
            _logger.exception(f"Error in after_run_callback: {e}")

    # ===== Agent Level Callbacks - invoke_agent span =====

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """
        Start Agent execution - create invoke_agent span.
        """
        try:
            # Extract conversation_id
            conversation_id = None

            conversation_id = self._session_id_from_callback_context(
                callback_context
            )

            # Create invocation object
            invocation = InvokeAgentInvocation(
                provider="google_adk",
                agent_name=agent.name,
            )

            # Set agent attributes
            if hasattr(agent, "id") and agent.id:
                invocation.agent_id = agent.id

            if hasattr(agent, "description") and agent.description:
                invocation.agent_description = agent.description

            if conversation_id:
                invocation.conversation_id = conversation_id

            user_id = getattr(callback_context, "user_id", None)
            if not user_id:
                user_id = getattr(
                    self._get_invocation_context(callback_context),
                    "user_id",
                    None,
                )
            if user_id:
                invocation.attributes["enduser.id"] = user_id

            # Start invocation (creates span)
            self._handler.start_invoke_agent(invocation)

            # Store invocation for later use
            agent_key = self._agent_key(agent, callback_context)
            self._active_agent_invocations[agent_key] = invocation

            _logger.debug(
                f"Started Agent invocation: invoke_agent {agent.name}"
            )

        except Exception as e:
            _logger.exception(f"Error in before_agent_callback: {e}")

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """
        End Agent execution - finish invoke_agent span.
        """
        try:
            agent_key = self._agent_key(agent, callback_context)
            invocation = self._active_agent_invocations.pop(agent_key, None)

            if invocation:
                # Stop invocation (ends span and records metrics automatically)
                self._handler.stop_invoke_agent(invocation)
                _logger.debug(f"Finished Agent invocation for {agent.name}")

            if self._is_root_agent(agent, callback_context):
                invocation_context = self._get_invocation_context(
                    callback_context
                )
                if invocation_context:
                    self._finish_runner_invocation(invocation_context)

        except Exception as e:
            _logger.exception(f"Error in after_agent_callback: {e}")

    # ===== LLM Level Callbacks - chat span =====

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """
        Start LLM call - create chat span.
        """
        try:
            # Extract model name
            model_name = llm_request.model if llm_request else "unknown"

            # Create invocation object
            invocation = LLMInvocation(
                request_model=model_name,
                provider=self._extractors._extract_provider_name(model_name),
            )

            # Extract input messages
            if llm_request.contents:
                input_messages = self._convert_contents_to_input_messages(
                    llm_request.contents
                )
                invocation.input_messages = input_messages

            # Extract request parameters
            if llm_request.config:
                config = llm_request.config
                max_tokens = self._get_real_attr(config, "max_tokens")
                if max_tokens:
                    invocation.max_tokens = max_tokens
                temperature = self._get_real_attr(config, "temperature")
                if temperature is not None:
                    invocation.temperature = temperature
                top_p = self._get_real_attr(config, "top_p")
                if top_p is not None:
                    invocation.top_p = top_p

            # Extract conversation_id and user_id
            session_id = self._session_id_from_callback_context(
                callback_context
            )
            if session_id:
                invocation.attributes["gen_ai.conversation.id"] = session_id

            user_id = getattr(callback_context, "user_id", None)
            if not user_id:
                user_id = getattr(
                    self._get_invocation_context(callback_context),
                    "user_id",
                    None,
                )
            if user_id:
                invocation.attributes["enduser.id"] = user_id

            # Start invocation (creates span)
            self._handler.start_llm(invocation)
            self._detach_current_context(invocation)

            # Store invocation for later use
            request_key = self._llm_key(callback_context, llm_request)
            self._active_llm_invocations[request_key] = invocation
            self._llm_context_tokens[request_key] = (
                _ACTIVE_LLM_REQUEST_KEY.set(request_key)
            )

            # Store the requested model for reliable retrieval later
            if hasattr(llm_request, "model") and llm_request.model:
                self._llm_req_models[request_key] = llm_request.model

            _logger.debug(f"Started LLM invocation: chat {model_name}")

        except Exception as e:
            _logger.exception(f"Error in before_model_callback: {e}")

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        """
        End LLM call - finish chat span.
        """
        try:
            request_key, llm_invocation = self._find_active_llm_invocation(
                callback_context
            )

            if llm_invocation:
                # Update invocation with response data
                if llm_response:
                    self._update_llm_invocation_from_response(
                        llm_invocation, llm_response, request_key
                    )

                    if self._is_streaming_partial_response(llm_response):
                        if llm_invocation.monotonic_first_token_s is None:
                            llm_invocation.monotonic_first_token_s = (
                                timeit.default_timer()
                            )
                        _logger.debug(
                            "Captured partial LLM response for %s",
                            request_key,
                        )
                        return None

                if request_key:
                    self._active_llm_invocations.pop(request_key, None)

                # Stop invocation (ends span and records metrics automatically)
                self._handler.stop_llm(llm_invocation)
                if request_key:
                    self._reset_active_llm_request_key(request_key)

                model_name = self._resolve_model_name(
                    llm_response, request_key, llm_invocation
                )
                if request_key:
                    self._llm_stream_outputs.pop(request_key, None)
                _logger.debug(
                    f"Finished LLM invocation for model {model_name}"
                )

        except Exception as e:
            _logger.exception(f"Error in after_model_callback: {e}")

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        """
        Handle LLM call errors.
        """
        try:
            # Find and finish the invocation with error status
            request_key, invocation = self._find_active_llm_invocation(
                callback_context, llm_request
            )
            if request_key and invocation:
                self._active_llm_invocations.pop(request_key, None)
                self._llm_stream_outputs.pop(request_key, None)
                self._reset_active_llm_request_key(request_key)

                # Fail invocation (sets error attributes and ends span)
                self._handler.fail_llm(
                    invocation, Error(message=str(error), type=type(error))
                )

            _logger.debug(f"Handled LLM error: {error}")

        except Exception as e:
            _logger.exception(f"Error in on_model_error_callback: {e}")

        return None

    # ===== Tool Level Callbacks - execute_tool span =====

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> None:
        """
        Start Tool execution - create execute_tool span.
        """
        try:
            # Create invocation object
            invocation = ExecuteToolInvocation(
                tool_name=tool.name,
                provider="google_adk",
            )

            # Set tool attributes
            if hasattr(tool, "description") and tool.description:
                invocation.tool_description = tool.description

            invocation.tool_type = "function"

            if hasattr(tool_context, "call_id") and tool_context.call_id:
                invocation.tool_call_id = tool_context.call_id

            # Set tool arguments (content capture is controlled by the util layer)
            if tool_args:
                invocation.tool_call_arguments = tool_args

            # Start invocation (creates span)
            self._handler.start_execute_tool(invocation)

            # Store invocation for later use
            tool_key = self._tool_key(tool, tool_args, tool_context)
            self._active_tool_invocations[tool_key] = invocation

            _logger.debug(f"Started Tool invocation: execute_tool {tool.name}")

        except Exception as e:
            _logger.exception(f"Error in before_tool_callback: {e}")

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> None:
        """
        End Tool execution - finish execute_tool span.
        """
        try:
            tool_key = self._tool_key(tool, tool_args, tool_context)
            invocation = self._active_tool_invocations.pop(tool_key, None)

            if invocation:
                # Set tool result (content capture is controlled by the util layer)
                if result:
                    invocation.tool_call_result = result

                # Stop invocation (ends span and records metrics automatically)
                self._handler.stop_execute_tool(invocation)
                _logger.debug(f"Finished Tool invocation for {tool.name}")

        except Exception as e:
            _logger.exception(f"Error in after_tool_callback: {e}")

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[dict]:
        """
        Handle Tool execution errors.
        """
        try:
            tool_key = self._tool_key(tool, tool_args, tool_context)
            invocation = self._active_tool_invocations.pop(tool_key, None)

            if invocation:
                # Fail invocation (sets error attributes and ends span)
                self._handler.fail_execute_tool(
                    invocation, Error(message=str(error), type=type(error))
                )

            _logger.debug(f"Handled Tool error: {error}")

        except Exception as e:
            _logger.exception(f"Error in on_tool_error_callback: {e}")

        return None

    # ===== Helper Methods =====

    @staticmethod
    def _detach_current_context(invocation: LLMInvocation) -> None:
        if invocation.context_token is None:
            return
        _safe_detach(invocation.context_token)

    def _reset_active_llm_request_key(self, request_key: str) -> None:
        token = self._llm_context_tokens.pop(request_key, None)
        if token is not None:
            try:
                _ACTIVE_LLM_REQUEST_KEY.reset(token)
                return
            except (RuntimeError, ValueError):
                pass

        if _ACTIVE_LLM_REQUEST_KEY.get() == request_key:
            _ACTIVE_LLM_REQUEST_KEY.set(None)

    @staticmethod
    def _get_invocation_context(
        callback_context: CallbackContext,
    ) -> Optional[InvocationContext]:
        return getattr(callback_context, "_invocation_context", None)

    def _finish_runner_invocation(
        self, invocation_context: InvocationContext
    ) -> None:
        runner_key = self._runner_key(invocation_context)
        invocation = self._active_runner_invocations.pop(runner_key, None)

        if invocation:
            self._handler.stop_invoke_agent(invocation)
            _logger.debug(
                "Finished Runner invocation for %s",
                getattr(invocation_context, "app_name", "unknown"),
            )

        self._runner_inputs.pop(runner_key, None)
        self._runner_outputs.pop(runner_key, None)

    def _is_root_agent(
        self, agent: BaseAgent, callback_context: CallbackContext
    ) -> bool:
        invocation_context = self._get_invocation_context(callback_context)
        if not invocation_context:
            return False

        root_agent = getattr(invocation_context, "agent", None)
        if root_agent is agent:
            return True

        root_name = getattr(root_agent, "name", None)
        return bool(root_name and root_name == getattr(agent, "name", None))

    @staticmethod
    def _is_root_final_event(
        event: Event, invocation_context: InvocationContext
    ) -> bool:
        is_final_response = getattr(event, "is_final_response", None)
        if callable(is_final_response):
            try:
                if not is_final_response():
                    return False
            except Exception:
                return False
        else:
            return False

        root_agent = getattr(invocation_context, "agent", None)
        root_name = getattr(root_agent, "name", None)
        event_author = getattr(event, "author", None)
        return bool(root_name and event_author and event_author == root_name)

    @staticmethod
    def _session_id_from_invocation_context(
        invocation_context: InvocationContext,
    ) -> Optional[str]:
        session = getattr(invocation_context, "session", None)
        return getattr(session, "id", None)

    def _session_id_from_callback_context(
        self, callback_context: CallbackContext
    ) -> Optional[str]:
        invocation_context = self._get_invocation_context(callback_context)
        if not invocation_context:
            return None
        return self._session_id_from_invocation_context(invocation_context)

    @staticmethod
    def _invocation_id_from_invocation_context(
        invocation_context: InvocationContext,
    ) -> str:
        invocation_id = getattr(invocation_context, "invocation_id", None)
        return str(invocation_id) if invocation_id is not None else "unknown"

    def _invocation_id_from_callback_context(
        self, callback_context: CallbackContext
    ) -> str:
        invocation_context = self._get_invocation_context(callback_context)
        if not invocation_context:
            return "unknown"
        return self._invocation_id_from_invocation_context(invocation_context)

    def _runner_key(self, invocation_context: InvocationContext) -> str:
        invocation_id = self._invocation_id_from_invocation_context(
            invocation_context
        )
        return f"runner_{invocation_id}"

    def _agent_key(
        self, agent: BaseAgent, callback_context: CallbackContext
    ) -> str:
        invocation_id = self._invocation_id_from_callback_context(
            callback_context
        )
        conversation_id = self._session_id_from_callback_context(
            callback_context
        )
        return f"agent_{invocation_id}_{id(agent)}_{conversation_id}"

    def _llm_key(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> str:
        invocation_id = self._invocation_id_from_callback_context(
            callback_context
        )
        session_id = self._session_id_from_callback_context(callback_context)
        return f"llm_{invocation_id}_{id(llm_request)}_{session_id}"

    def _tool_key(
        self,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> str:
        invocation_context = getattr(tool_context, "_invocation_context", None)
        invocation_id = (
            self._invocation_id_from_invocation_context(invocation_context)
            if invocation_context
            else str(getattr(tool_context, "invocation_id", "unknown"))
        )
        call_id = getattr(tool_context, "call_id", None)
        return f"tool_{invocation_id}_{call_id}_{id(tool)}_{id(tool_args)}"

    def _find_active_llm_invocation(
        self,
        callback_context: CallbackContext,
        llm_request: Optional[LlmRequest] = None,
    ) -> tuple[Optional[str], Optional[LLMInvocation]]:
        context_request_key = _ACTIVE_LLM_REQUEST_KEY.get()
        if context_request_key:
            invocation = self._active_llm_invocations.get(context_request_key)
            if invocation:
                return context_request_key, invocation

        if llm_request is not None:
            request_key = self._llm_key(callback_context, llm_request)
            invocation = self._active_llm_invocations.get(request_key)
            if invocation:
                return request_key, invocation

        invocation_id = self._invocation_id_from_callback_context(
            callback_context
        )
        session_id = self._session_id_from_callback_context(callback_context)
        preferred_prefix = f"llm_{invocation_id}_"

        for key, invocation in list(self._active_llm_invocations.items()):
            if key.startswith(preferred_prefix):
                return key, invocation

        for key, invocation in list(self._active_llm_invocations.items()):
            if (
                key.startswith("llm_")
                and session_id
                and key.endswith(f"_{session_id}")
            ):
                return key, invocation

        return None, None

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        """
        Extract text from ADK content objects.

        Handles various content types: plain strings, Content objects with
        parts/text attributes, and other objects (converted via str()).

        Args:
            content: Content object (could be types.Content, string, etc.)

        Returns:
            Extracted text string
        """
        if not content:
            return ""
        if isinstance(content, str):
            return content
        if hasattr(content, "parts") and content.parts:
            text_parts = []
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
            return "".join(text_parts)
        if hasattr(content, "text"):
            return content.text or ""
        return str(content)

    @staticmethod
    def _is_mock_placeholder(value: Any) -> bool:
        return type(value).__module__.startswith("unittest.mock")

    @staticmethod
    def _mock_has_explicit_attrs(
        value: Any, attr_names: tuple[str, ...]
    ) -> bool:
        value_dict = getattr(value, "__dict__", {})
        return any(attr_name in value_dict for attr_name in attr_names)

    def _get_real_attr(self, value: Any, attr_name: str) -> Any:
        attr_value = getattr(value, attr_name, None)
        if self._is_mock_placeholder(attr_value):
            return None
        return attr_value

    def _resolve_model_name(
        self,
        llm_response: LlmResponse,
        request_key: Optional[str],
        invocation: LLMInvocation,
    ) -> str:
        """
        Resolve model name with robust fallbacks.

        Args:
            llm_response: LLM response object
            request_key: Request key for stored models, if known
            invocation: LLMInvocation object

        Returns:
            Model name string
        """
        model_name = None

        # 1) Prefer response model fields if available
        if llm_response:
            model_name = self._get_response_model_name(llm_response)

        # 2) Use stored request model by request_key
        if (
            not model_name
            and request_key
            and request_key in self._llm_req_models
        ):
            model_name = self._llm_req_models.pop(request_key, None)

        # 3) Use invocation request_model
        if not model_name and invocation and invocation.request_model:
            model_name = invocation.request_model

        # 4) Final fallback
        if not model_name:
            model_name = "unknown"

        return model_name

    @staticmethod
    def _get_response_model_name(llm_response: LlmResponse) -> Optional[str]:
        for attr_name in ("model", "model_version", "modelVersion"):
            model_name = getattr(llm_response, attr_name, None)
            if (
                model_name
                and not GoogleAdkObservabilityPlugin._is_mock_placeholder(
                    model_name
                )
            ):
                return model_name
        return None

    @staticmethod
    def _is_streaming_partial_response(llm_response: LlmResponse) -> bool:
        return bool(getattr(llm_response, "partial", False)) and not bool(
            getattr(llm_response, "turn_complete", False)
        )

    def _merge_stream_output(self, request_key: str, text: str) -> str:
        if not text:
            return self._llm_stream_outputs.get(request_key, "")

        # ADK streaming responses are cumulative snapshots, not deltas.
        merged = text
        self._llm_stream_outputs[request_key] = merged
        return merged

    def _update_llm_invocation_from_response(
        self,
        invocation: LLMInvocation,
        llm_response: LlmResponse,
        request_key: Optional[str],
    ) -> None:
        response_model = self._get_response_model_name(llm_response)
        if response_model:
            invocation.response_model_name = response_model

        usage = getattr(llm_response, "usage_metadata", None)
        if self._is_mock_placeholder(
            usage
        ) and not self._mock_has_explicit_attrs(
            usage,
            ("prompt_token_count", "candidates_token_count"),
        ):
            usage = None
        if usage:
            input_tokens = self._get_real_attr(usage, "prompt_token_count")
            if input_tokens is not None:
                invocation.input_tokens = input_tokens

            output_tokens = self._get_real_attr(
                usage, "candidates_token_count"
            )
            if output_tokens is not None:
                invocation.output_tokens = output_tokens

        finish_reason = self._get_real_attr(llm_response, "finish_reason")
        if finish_reason:
            if hasattr(finish_reason, "value"):
                finish_reason = finish_reason.value
            elif not isinstance(finish_reason, (str, int, float, bool)):
                finish_reason = str(finish_reason)
            invocation.finish_reasons = [finish_reason]

        extracted_text = self._extract_text_from_llm_response(llm_response)
        accumulated_text = (
            self._merge_stream_output(request_key, extracted_text)
            if request_key
            else extracted_text
        )
        output_messages = self._convert_text_to_output_messages(
            accumulated_text,
            llm_response,
        )
        if output_messages:
            invocation.output_messages = output_messages

    def _convert_user_message_to_input_messages(
        self, user_message: types.Content
    ) -> List[InputMessage]:
        """
        Convert ADK user message to GenAI InputMessage format.

        Args:
            user_message: ADK Content object

        Returns:
            List of InputMessage objects
        """
        input_messages = []
        if (
            user_message
            and hasattr(user_message, "role")
            and hasattr(user_message, "parts")
        ):
            parts = []
            for part in user_message.parts:
                if hasattr(part, "text"):
                    parts.append(Text(content=part.text))
            if parts:
                input_messages.append(
                    InputMessage(role=user_message.role, parts=parts)
                )
        return input_messages

    def _convert_contents_to_input_messages(
        self, contents: List[types.Content]
    ) -> List[InputMessage]:
        """
        Convert ADK contents to GenAI InputMessage format.

        Args:
            contents: List of ADK Content objects

        Returns:
            List of InputMessage objects
        """
        input_messages = []
        for content in contents:
            if hasattr(content, "role") and hasattr(content, "parts"):
                parts = []
                for part in content.parts:
                    if hasattr(part, "text"):
                        parts.append(Text(content=part.text))
                if parts:
                    input_messages.append(
                        InputMessage(role=content.role, parts=parts)
                    )
        return input_messages

    def _extract_text_from_llm_response(
        self, llm_response: LlmResponse
    ) -> str:
        if not llm_response:
            return ""

        content = self._get_real_attr(llm_response, "content")
        if content is not None:
            return self._extract_text_from_content(content)

        text = self._get_real_attr(llm_response, "text")
        if text is not None:
            return self._extract_text_from_content(text)

        return ""

    def _convert_text_to_output_messages(
        self, text: str, llm_response: LlmResponse
    ) -> List[OutputMessage]:
        """
        Convert ADK response text to GenAI OutputMessage format.

        Args:
            text: ADK response text
            llm_response: ADK LlmResponse object, used for finish reason

        Returns:
            List of OutputMessage objects
        """
        output_messages = []

        if not llm_response:
            return output_messages

        try:
            # Extract finish reason
            finish_reason = (
                getattr(llm_response, "finish_reason", None) or "stop"
            )
            if hasattr(finish_reason, "value"):
                finish_reason = finish_reason.value
            elif not isinstance(finish_reason, (str, int, float, bool)):
                finish_reason = str(finish_reason)

            if text:
                output_messages.append(
                    OutputMessage(
                        role="assistant",
                        parts=[Text(content=text)],
                        finish_reason=finish_reason,
                    )
                )
        except Exception as e:
            _logger.debug(f"Failed to extract output messages: {e}")

        return output_messages

    def _convert_llm_response_to_output_messages(
        self, llm_response: LlmResponse
    ) -> List[OutputMessage]:
        """
        Convert ADK LlmResponse to GenAI OutputMessage format.

        Args:
            llm_response: ADK LlmResponse object

        Returns:
            List of OutputMessage objects
        """
        return self._convert_text_to_output_messages(
            self._extract_text_from_llm_response(llm_response),
            llm_response,
        )
