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

"""Message format converter for AgentScope instrumentation.

This module provides utilities to convert messages from different AI model providers
into a unified format for OpenTelemetry instrumentation and frontend display.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

logger = logging.getLogger(__name__)


def _create_text_part_dict(content: str) -> Dict[str, Any]:
    """Create text part as dict"""
    return {"type": "text", "content": content}


def _create_tool_call_part_dict(
    name: str,
    call_id: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create tool call part as dict"""
    part = {"type": "tool_call", "name": name}
    if call_id is not None:
        part["id"] = call_id
    if arguments is not None:
        part["arguments"] = arguments
    return part


def _create_tool_response_part_dict(
    response: Any,
    call_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create tool response part as dict"""
    part = {"type": "tool_call_response", "response": response}
    if call_id is not None:
        part["id"] = call_id
    return part


def _create_generic_part_dict(part_type: str, **kwargs: Any) -> Dict[str, Any]:
    """Create generic part as dict"""
    part = {"type": part_type}
    part.update(kwargs)
    return part


class BaseMessageParser(ABC):
    """Base class for message parsers"""

    @abstractmethod
    def parse_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Parse single message to unified dict format with role and parts"""
        pass

    def parse_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse message list to unified dict format"""
        parsed_messages = []
        for msg in messages:
            try:
                parsed_msg = self.parse_message(msg)
                if parsed_msg.get("parts"):  # Only add messages with content
                    parsed_messages.append(parsed_msg)
            except Exception as e:
                logger.warning(f"Failed to parse message: {e}, message: {msg}")
                continue
        return parsed_messages

    def _get_role(self, msg: Dict[str, Any]) -> str:
        """Get role with role mapping support"""
        role = msg.get("role", "user")
        # Gemini special handling
        if role == "model":
            return "assistant"
        return role

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        """Safe JSON parsing"""
        if isinstance(text, dict):
            return text
        try:
            return json.loads(text) if isinstance(text, str) else {}
        except json.JSONDecodeError:
            return {}


class OpenAIMessageParser(BaseMessageParser):
    """OpenAIformat message parser"""

    def parse_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        role = self._get_role(msg)
        parts = []

        # Handle tool messages
        if role == "tool":
            part = _create_tool_response_part_dict(
                response=msg.get("content", ""),
                call_id=msg.get("tool_call_id", ""),
            )
            parts.append(part)
        else:
            # Handle tool_calls
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    function_info = tool_call.get("function", {})
                    arguments = self._safe_json_loads(
                        function_info.get("arguments", "{}")
                    )

                    part = _create_tool_call_part_dict(
                        name=function_info.get("name", "unknown_tool"),
                        call_id=tool_call.get("id", "unknown_id"),
                        arguments=arguments,
                    )
                    parts.append(part)

            # Handle content
            content = msg.get("content")
            if isinstance(content, str):
                if content.strip():
                    parts.append(_create_text_part_dict(content))
            elif isinstance(content, list):
                parts.extend(self._parse_content_list(content))

        # Add empty text part if no parts and no tool_calls
        if not parts and not msg.get("tool_calls"):
            parts.append(_create_text_part_dict(""))

        return {"role": role, "parts": parts}

    def _parse_content_list(
        self, content_list: List[Any]
    ) -> List[Dict[str, Any]]:
        """Parse content list"""
        parts = []
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get("type", "text")
                if content_type == "text":
                    text_content = item.get("text", "")
                    if text_content:
                        parts.append(_create_text_part_dict(text_content))
                elif content_type == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url:
                        parts.append(
                            _create_generic_part_dict(
                                "image", content=image_url
                            )
                        )
                elif content_type == "input_audio":
                    audio_data = item.get("input_audio", {}).get("data", "")
                    if audio_data:
                        parts.append(
                            _create_generic_part_dict(
                                "audio", content=audio_data
                            )
                        )
                else:
                    parts.append(
                        _create_generic_part_dict(
                            content_type,
                            content=str(item),
                            hint="unsupported content type",
                        )
                    )
            elif isinstance(item, str):
                parts.append(_create_text_part_dict(item))
        return parts


class AnthropicMessageParser(BaseMessageParser):
    """Anthropicformat message parser"""

    def parse_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        role = self._get_role(msg)
        parts = []

        content = msg.get("content", [])
        if isinstance(content, str):
            parts.append(_create_text_part_dict(content))
        elif isinstance(content, list):
            parts.extend(self._parse_content_list(content))

        if not parts:
            parts.append(_create_text_part_dict(""))

        return {"role": role, "parts": parts}

    def _parse_content_list(
        self, content_list: List[Any]
    ) -> List[Dict[str, Any]]:
        """Parse Anthropic content list"""
        parts = []
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get("type", "text")
                if content_type == "text":
                    text_content = item.get("text")
                    if not text_content:
                        continue
                    parts.append(_create_text_part_dict(text_content))
                elif content_type == "tool_use":
                    parts.append(
                        _create_tool_call_part_dict(
                            name=item.get("name", ""),
                            call_id=item.get("id", ""),
                            arguments=item.get("input", {}),
                        )
                    )
                elif content_type == "tool_result":
                    result_text = self._extract_tool_result(
                        item.get("content", [])
                    )
                    parts.append(
                        _create_tool_response_part_dict(
                            response=result_text,
                            call_id=item.get("tool_use_id", ""),
                        )
                    )
                elif content_type == "image":
                    source = item.get("source", {})
                    if source.get("type") == "url":
                        parts.append(
                            _create_generic_part_dict(
                                "image", content=source.get("url", "")
                            )
                        )
                    elif source.get("type") == "base64":
                        content_url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                        parts.append(
                            _create_generic_part_dict(
                                "image", content=content_url
                            )
                        )
                else:
                    parts.append(
                        _create_generic_part_dict(
                            content_type,
                            content=str(item),
                            hint="unsupported content type",
                        )
                    )
            elif isinstance(item, str):
                parts.append(_create_text_part_dict(item))
        return parts

    def _extract_tool_result(self, content_blocks: List[Any]) -> str:
        """Extract tool call result"""
        if isinstance(content_blocks, list):
            result_text = ""
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    result_text += block.get("text", "")
            return result_text
        return str(content_blocks)


class GeminiMessageParser(BaseMessageParser):
    """Geminiformat message parser"""

    def parse_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        role = self._get_role(msg)
        parts = []

        parts_list = msg.get("parts", [])
        if isinstance(parts_list, list):
            parts.extend(self._parse_parts_list(parts_list))

        if not parts:
            parts.append(_create_text_part_dict(""))

        return {"role": role, "parts": parts}

    def _parse_parts_list(self, parts_list: List[Any]) -> List[Dict[str, Any]]:
        """Parse Gemini parts list"""
        parts = []
        for item in parts_list:
            if isinstance(item, dict):
                if "text" in item:
                    if not item["text"]:
                        continue
                    parts.append(_create_text_part_dict(item["text"]))
                elif "function_call" in item:
                    func_call = item["function_call"]
                    parts.append(
                        _create_tool_call_part_dict(
                            name=func_call.get("name", ""),
                            call_id=func_call.get("id", ""),
                            arguments=func_call.get("args", {}),
                        )
                    )
                elif "function_response" in item:
                    func_resp = item["function_response"]
                    response_content = func_resp.get("response", {})
                    result_text = (
                        response_content.get("output", "")
                        if isinstance(response_content, dict)
                        else str(response_content)
                    )
                    parts.append(
                        _create_tool_response_part_dict(
                            response=result_text,
                            call_id=func_resp.get("id", ""),
                        )
                    )
                elif "inline_data" in item:
                    parts.append(self._parse_inline_data(item["inline_data"]))
                else:
                    parts.append(
                        _create_generic_part_dict(
                            "unknown",
                            content=str(item),
                            hint="unsupported part type",
                        )
                    )
            elif isinstance(item, str):
                parts.append(_create_text_part_dict(item))
        return parts

    def _parse_inline_data(
        self, inline_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse inline data"""
        mime_type = inline_data.get("mime_type", "")
        data = inline_data.get("data", "")

        if mime_type.startswith("image/"):
            return _create_generic_part_dict(
                "image", content=f"data:{mime_type};base64,{data}"
            )
        elif mime_type.startswith("audio/"):
            return _create_generic_part_dict("audio", content=data)
        else:
            return _create_generic_part_dict(
                "unknown",
                content=str(inline_data),
                hint="unsupported inline_data type",
            )


class DashScopeMessageParser(BaseMessageParser):
    """DashScopeformat message parser"""

    def parse_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        role = self._get_role(msg)
        parts = []

        # Handle tool messages
        if role == "tool":
            parts.append(
                _create_tool_response_part_dict(
                    response=msg.get("content", ""),
                    call_id=msg.get("tool_call_id", ""),
                )
            )
        else:
            # Handle tool_calls
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    function_info = tool_call.get("function", {})
                    arguments = self._safe_json_loads(
                        function_info.get("arguments", "{}")
                    )

                    parts.append(
                        _create_tool_call_part_dict(
                            name=function_info.get("name", ""),
                            call_id=tool_call.get("id", ""),
                            arguments=arguments,
                        )
                    )

            # Handle content
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(_create_text_part_dict(content))
            elif isinstance(content, list):
                parts.extend(self._parse_content_list(content))

        if not parts and not msg.get("tool_calls"):
            parts.append(_create_text_part_dict(""))

        return {"role": role, "parts": parts}

    def _parse_content_list(
        self, content_list: List[Any]
    ) -> List[Dict[str, Any]]:
        """Parse DashScope content list"""
        parts = []
        for item in content_list:
            if isinstance(item, dict):
                if "text" in item:
                    if not item["text"]:
                        continue
                    parts.append(_create_text_part_dict(item["text"]))
                elif "image" in item:
                    parts.append(
                        _create_generic_part_dict(
                            "image", content=item["image"]
                        )
                    )
                elif "audio" in item:
                    parts.append(
                        _create_generic_part_dict(
                            "audio", content=item["audio"]
                        )
                    )
                else:
                    parts.append(
                        _create_generic_part_dict(
                            "unknown",
                            content=str(item),
                            hint="unsupported content type",
                        )
                    )
            elif isinstance(item, str):
                parts.append(_create_text_part_dict(item))
        return parts


class OllamaMessageParser(BaseMessageParser):
    """Ollamaformat message parser"""

    def parse_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        role = self._get_role(msg)
        parts = []

        # Handle tool messages
        if role == "tool":
            parts.append(
                _create_tool_response_part_dict(
                    response=msg.get("content", ""),
                    call_id=msg.get("tool_call_id", ""),
                )
            )
        else:
            # Handle tool_calls
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    function_info = tool_call.get("function", {})
                    parts.append(
                        _create_tool_call_part_dict(
                            name=function_info.get("name", ""),
                            call_id=tool_call.get("id", ""),
                            arguments=function_info.get("arguments", {}),
                        )
                    )

            # Handle text content
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(_create_text_part_dict(content))
            elif content is None and not msg.get("tool_calls"):
                parts.append(_create_text_part_dict(""))

            # Handle images (Ollama-specific images field)
            if "images" in msg and msg["images"]:
                for image_data in msg["images"]:
                    if isinstance(image_data, str):
                        parts.append(
                            _create_generic_part_dict(
                                "image",
                                content=f"data:image/png;base64,{image_data}",
                            )
                        )
                    else:
                        parts.append(
                            _create_generic_part_dict(
                                "image", content=str(image_data)
                            )
                        )

        if not parts and not msg.get("tool_calls"):
            parts.append(_create_text_part_dict(""))

        return {"role": role, "parts": parts}


class DeepSeekMessageParser(BaseMessageParser):
    """DeepSeekformat message parser"""

    def parse_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        role = self._get_role(msg)
        parts = []

        # Handle tool messages
        if role == "tool":
            parts.append(
                _create_tool_response_part_dict(
                    response=msg.get("content", ""),
                    call_id=msg.get("tool_call_id", ""),
                )
            )
        else:
            # Handle tool_calls
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    function_info = tool_call.get("function", {})
                    arguments = self._safe_json_loads(
                        function_info.get("arguments", "{}")
                    )

                    parts.append(
                        _create_tool_call_part_dict(
                            name=function_info.get("name", ""),
                            call_id=tool_call.get("id", ""),
                            arguments=arguments,
                        )
                    )

            # Handle content
            content = msg.get("content")
            if isinstance(content, str) and content and content.strip():
                parts.append(_create_text_part_dict(content))
            elif content is None and msg.get("tool_calls"):
                # DeepSeekcontent may be None when tool_calls present
                pass  # No need to add empty text
            elif isinstance(content, list):
                parts.extend(self._parse_content_list(content))

        if not parts and not msg.get("tool_calls"):
            parts.append(_create_text_part_dict(""))

        return {"role": role, "parts": parts}

    def _parse_content_list(
        self, content_list: List[Any]
    ) -> List[Dict[str, Any]]:
        """Parse DeepSeek content list"""
        parts = []
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get("type", "text")
                if content_type == "text":
                    text_content = item.get("text")
                    if not text_content:
                        continue
                    parts.append(_create_text_part_dict(text_content))
                else:
                    parts.append(
                        _create_generic_part_dict(
                            content_type,
                            content=str(item),
                            hint="unsupported content type",
                        )
                    )
            elif isinstance(item, str):
                parts.append(_create_text_part_dict(item))
        return parts


class DefaultMessageParser(BaseMessageParser):
    """Default message parser for unknown formats"""

    def parse_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        role = self._get_role(msg)
        parts = []

        # Try to extract text from content field
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(_create_text_part_dict(content))
        elif isinstance(content, list):
            # Try to parse list format content
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        if not item["text"]:
                            continue
                        parts.append(_create_text_part_dict(str(item["text"])))
                    else:
                        parts.append(
                            _create_generic_part_dict(
                                "unknown",
                                content=str(item),
                                hint="unknown format",
                            )
                        )
                else:
                    parts.append(_create_text_part_dict(str(item)))
        else:
            parts.append(_create_text_part_dict(str(msg)))

        if not parts:
            parts.append(_create_text_part_dict(""))

        return {"role": role, "parts": parts}


def get_message_converter(
    provide_name: Optional[str],
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Return appropriate message format converter based on model type。

    Args:
        provide_name: AI model provider

    Returns:
        Corresponding message conversion function with signature:
        (messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]

        Converted unified format complies with JSON Schema standard:
        [
            {
                "role": "user|assistant|tool|system",
                "parts": [
                    {
                        "type": "text",
                        "content": "..."
                    },
                    {
                        "type": "tool_call",
                        "name": "...",
                        "id": "...",
                        "arguments": {...}
                    },
                    {
                        "type": "tool_call_response",
                        "response": "...",
                        "id": "..."
                    }
                ]
            }
        ]
    """
    # Select corresponding parser
    if provide_name == GenAIAttributes.GenAiProviderNameValues.OPENAI.value:
        parser = OpenAIMessageParser()
    elif (
        provide_name == GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value
    ):
        parser = AnthropicMessageParser()
    elif (
        provide_name
        == GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value
    ):
        parser = GeminiMessageParser()
    elif (
        provide_name == GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value
    ):
        parser = DeepSeekMessageParser()
    elif (
        provide_name == "dashscope"
    ):  # AgentScopeGenAiProviderName.DASHSCOPE.value
        parser = DashScopeMessageParser()
    elif provide_name == "ollama":  # AgentScopeGenAiProviderName.OLLAMA.value
        parser = OllamaMessageParser()
    elif (
        provide_name == "moonshot_ai"
    ):  # AgentScopeGenAiProviderName.MOONSHOT.value
        # Moonshot uses OpenAI-compatible API format
        parser = OpenAIMessageParser()
    else:
        parser = DefaultMessageParser()

    return parser.parse_messages
