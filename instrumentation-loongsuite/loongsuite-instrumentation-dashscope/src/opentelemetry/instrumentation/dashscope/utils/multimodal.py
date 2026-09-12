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

"""Utility functions for DashScope multimodal API instrumentation.

This module contains utilities for:
- ImageSynthesis
- MultiModalConversation
- VideoSynthesis
- SpeechSynthesizer (V1 and V2)
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from opentelemetry.util.genai.types import (
    Blob,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    Uri,
)

from .common import _extract_usage, _get_parameter
from .generation import _extract_tool_definitions

logger = logging.getLogger(__name__)

# ============================================================================
# ImageSynthesis utilities
# ============================================================================


def _create_invocation_from_image_synthesis(
    kwargs: dict, model: Optional[str] = None
) -> LLMInvocation:
    """Create LLMInvocation from ImageSynthesis.call or async_call kwargs.

    Args:
        kwargs: ImageSynthesis.call or async_call kwargs
        model: Model name (if not in kwargs)

    Returns:
        LLMInvocation object
    """
    request_model = kwargs.get("model") or model
    if not request_model:
        raise ValueError("Model name is required")

    invocation = LLMInvocation(request_model=request_model)
    invocation.provider = "dashscope"
    invocation.operation_name = "generate_content"

    # Extract prompt as input message
    prompt = kwargs.get("prompt")
    if prompt:
        if isinstance(prompt, str):
            invocation.input_messages = [
                InputMessage(
                    role="user",
                    parts=[Text(content=prompt, type="text")],
                )
            ]
        elif isinstance(prompt, list):
            # Handle list of prompts
            parts = []
            for p in prompt:
                if isinstance(p, str):
                    parts.append(Text(content=p, type="text"))
            if parts:
                invocation.input_messages = [
                    InputMessage(role="user", parts=parts)
                ]

    return invocation


def _update_invocation_from_image_synthesis_response(
    invocation: LLMInvocation, response: Any
) -> None:
    """Update LLMInvocation with ImageSynthesis response data (for call() and wait()).

    Args:
        invocation: LLMInvocation to update
        response: ImageSynthesisResponse object
    """
    if not response:
        return

    try:
        # Extract token usage
        input_tokens, output_tokens = _extract_usage(response)
        invocation.input_tokens = input_tokens
        invocation.output_tokens = output_tokens

        # Extract response model name (if available)
        try:
            response_model = getattr(response, "model", None)
            if response_model:
                invocation.response_model_name = response_model
        except (KeyError, AttributeError) as e:
            logger.debug(
                "Failed to extract response model from ImageSynthesis response: %s",
                e,
            )

        # Extract task_id from output and set as response_id
        # Note: For ImageSynthesis, response_id should be task_id, not request_id
        try:
            output = getattr(response, "output", None)
            if output:
                # Extract task_id
                task_id = None
                if hasattr(output, "get"):
                    task_id = output.get("task_id")
                elif hasattr(output, "task_id"):
                    task_id = getattr(output, "task_id", None)

                if task_id:
                    invocation.response_id = task_id

                # Extract image URLs from results and add as Uri MessageParts
                results = None
                if hasattr(output, "get"):
                    results = output.get("results")
                elif hasattr(output, "results"):
                    results = getattr(output, "results", None)

                if results and isinstance(results, list):
                    image_uris = []
                    for result in results:
                        if isinstance(result, dict):
                            url = result.get("url")
                            if url:
                                image_uris.append(
                                    Uri(
                                        uri=url,
                                        modality="image",
                                        mime_type=None,
                                        type="uri",
                                    )
                                )
                        elif hasattr(result, "url"):
                            url = getattr(result, "url", None)
                            if url:
                                image_uris.append(
                                    Uri(
                                        uri=url,
                                        modality="image",
                                        mime_type=None,
                                        type="uri",
                                    )
                                )
                    if image_uris:
                        # Add image URIs to output messages
                        # If output_messages is empty, create a new one
                        if not invocation.output_messages:
                            invocation.output_messages = [
                                OutputMessage(
                                    role="assistant",
                                    parts=image_uris,
                                    finish_reason="stop",
                                )
                            ]
                        else:
                            # Append URIs to the last output message
                            invocation.output_messages[-1].parts.extend(
                                image_uris
                            )
        except (KeyError, AttributeError) as e:
            logger.debug(
                "Failed to extract image URLs from ImageSynthesis response: %s",
                e,
            )
    except (KeyError, AttributeError) as e:
        # If any attribute access fails, silently continue with available data
        logger.debug(
            "Failed to update invocation from ImageSynthesis response: %s", e
        )


def _update_invocation_from_image_synthesis_async_response(
    invocation: LLMInvocation, response: Any
) -> None:
    """Update LLMInvocation with ImageSynthesis async_call response data.

    This is called when async_call() returns, before wait() is called.
    Extracts task_id and sets it as response_id.

    Args:
        invocation: LLMInvocation to update
        response: ImageSynthesisResponse object from async_call()
    """
    if not response:
        return

    try:
        # Extract task_id from output and set as response_id
        output = getattr(response, "output", None)
        if output:
            task_id = None
            if hasattr(output, "get"):
                task_id = output.get("task_id")
            elif hasattr(output, "task_id"):
                task_id = getattr(output, "task_id", None)

            if task_id:
                invocation.response_id = task_id
    except (KeyError, AttributeError) as e:
        logger.debug(
            "Failed to extract task_id from ImageSynthesis async_call response: %s",
            e,
        )


# ============================================================================
# MultiModalConversation utilities
# ============================================================================


def _extract_multimodal_input_messages(kwargs: dict) -> List[InputMessage]:
    """Extract input messages from MultiModalConversation API kwargs.

    MultiModalConversation uses a different message format than Generation.
    Messages can contain text, image URLs, audio URLs, video URLs, etc.
    """
    input_messages = []

    # Check for messages format
    messages = kwargs.get("messages")
    if not messages:
        return input_messages

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", [])

            parts = []

            # MultiModal content is always a list of content items
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        # Text content
                        if "text" in item:
                            parts.append(
                                Text(content=item["text"], type="text")
                            )
                        # Image content
                        elif "image" in item:
                            parts.append(
                                Uri(
                                    uri=item["image"],
                                    modality="image",
                                    mime_type=None,
                                    type="uri",
                                )
                            )
                        # Audio content
                        elif "audio" in item:
                            parts.append(
                                Uri(
                                    uri=item["audio"],
                                    modality="audio",
                                    mime_type=None,
                                    type="uri",
                                )
                            )
                        # Video content
                        elif "video" in item:
                            parts.append(
                                Uri(
                                    uri=item["video"],
                                    modality="video",
                                    mime_type=None,
                                    type="uri",
                                )
                            )
            elif isinstance(content, str):
                # Simple text content
                parts.append(Text(content=content, type="text"))

            if parts:
                input_messages.append(InputMessage(role=role, parts=parts))

    return input_messages


def _extract_multimodal_output_messages(response: Any) -> List[OutputMessage]:
    """Extract output messages from MultiModalConversation response.

    Args:
        response: MultiModalConversationResponse object

    Returns:
        List of OutputMessage objects
    """
    output_messages = []

    if not response:
        return output_messages

    try:
        output = getattr(response, "output", None)
        if not output:
            return output_messages

        # Check for choices format
        choices = getattr(output, "choices", None)
        if choices and isinstance(choices, list) and len(choices) > 0:
            for choice in choices:
                if not choice:
                    continue

                message = getattr(choice, "message", None)
                if not message:
                    continue

                content = getattr(message, "content", None)
                finish_reason = getattr(choice, "finish_reason", "stop")

                parts = []

                # Handle content
                if content:
                    if isinstance(content, str):
                        parts.append(Text(content=content, type="text"))
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                # Text content
                                if "text" in item:
                                    parts.append(
                                        Text(
                                            content=item["text"],
                                            type="text",
                                        )
                                    )
                                # Image content
                                elif "image" in item:
                                    parts.append(
                                        Uri(
                                            uri=item["image"],
                                            modality="image",
                                            mime_type=None,
                                            type="uri",
                                        )
                                    )
                                # Audio content (when modalities includes "audio")
                                elif "audio" in item:
                                    parts.append(
                                        Uri(
                                            uri=item["audio"],
                                            modality="audio",
                                            mime_type=None,
                                            type="uri",
                                        )
                                    )
                                # Video content
                                elif "video" in item:
                                    parts.append(
                                        Uri(
                                            uri=item["video"],
                                            modality="video",
                                            mime_type=None,
                                            type="uri",
                                        )
                                    )
                            elif isinstance(item, str):
                                parts.append(Text(content=item, type="text"))

                if parts:
                    output_messages.append(
                        OutputMessage(
                            role="assistant",
                            parts=parts,
                            finish_reason=finish_reason or "stop",
                        )
                    )

        # Fallback to text format
        else:
            text = getattr(output, "text", None)
            if text:
                output_messages.append(
                    OutputMessage(
                        role="assistant",
                        parts=[Text(content=text, type="text")],
                        finish_reason="stop",
                    )
                )

    except (KeyError, AttributeError) as e:
        logger.debug(
            "Failed to extract output messages from MultiModalConversation response: %s",
            e,
        )

    return output_messages


def _create_invocation_from_multimodal_conversation(
    kwargs: dict, model: Optional[str] = None
) -> LLMInvocation:
    """Create LLMInvocation from MultiModalConversation.call kwargs.

    Args:
        kwargs: MultiModalConversation.call kwargs
        model: Model name (if not in kwargs)

    Returns:
        LLMInvocation object
    """
    request_model = kwargs.get("model") or model
    if not request_model:
        raise ValueError("Model name is required")

    invocation = LLMInvocation(request_model=request_model)
    invocation.provider = "dashscope"
    invocation.input_messages = _extract_multimodal_input_messages(kwargs)

    # Extract tool definitions (if present)
    # MultiModalConversation supports Function Calling via tools parameter
    invocation.tool_definitions = _extract_tool_definitions(kwargs)

    # Extract parameters
    # Temperature
    temperature = _get_parameter(kwargs, "temperature")
    if temperature is not None:
        invocation.attributes["gen_ai.request.temperature"] = temperature

    # Top-p
    top_p = _get_parameter(kwargs, "top_p")
    if top_p is not None:
        invocation.attributes["gen_ai.request.top_p"] = top_p

    # Top-k
    top_k = _get_parameter(kwargs, "top_k")
    if top_k is not None:
        invocation.attributes["gen_ai.request.top_k"] = top_k

    # Max tokens
    max_tokens = _get_parameter(kwargs, "max_tokens")
    if max_tokens is not None:
        invocation.attributes["gen_ai.request.max_tokens"] = max_tokens

    # Seed
    seed = _get_parameter(kwargs, "seed")
    if seed is not None:
        invocation.attributes["gen_ai.request.seed"] = seed

    return invocation


def _update_invocation_from_multimodal_response(
    invocation: LLMInvocation, response: Any
) -> None:
    """Update LLMInvocation with MultiModalConversation response data.

    Args:
        invocation: LLMInvocation to update
        response: MultiModalConversationResponse object
    """
    if not response:
        return

    try:
        # Extract output messages
        invocation.output_messages = _extract_multimodal_output_messages(
            response
        )

        # Extract token usage
        input_tokens, output_tokens = _extract_usage(response)
        invocation.input_tokens = input_tokens
        invocation.output_tokens = output_tokens

        # Extract response model name (if available)
        try:
            response_model = getattr(response, "model", None)
            if response_model:
                invocation.response_model_name = response_model
        except (KeyError, AttributeError) as e:
            logger.debug(
                "Failed to extract response model from MultiModalConversation response: %s",
                e,
            )

        # Extract request ID
        try:
            request_id = getattr(response, "request_id", None)
            if request_id:
                invocation.response_id = request_id
        except (KeyError, AttributeError) as e:
            logger.debug(
                "Failed to extract request_id from MultiModalConversation response: %s",
                e,
            )

    except (KeyError, AttributeError) as e:
        logger.debug(
            "Failed to update invocation from MultiModalConversation response: %s",
            e,
        )


# ============================================================================
# VideoSynthesis utilities
# ============================================================================


def _create_invocation_from_video_synthesis(
    kwargs: dict, model: Optional[str] = None
) -> LLMInvocation:
    """Create LLMInvocation from VideoSynthesis.call or async_call kwargs.

    Args:
        kwargs: VideoSynthesis.call or async_call kwargs
        model: Model name (if not in kwargs)

    Returns:
        LLMInvocation object
    """
    request_model = kwargs.get("model") or model
    if not request_model:
        raise ValueError("Model name is required")

    invocation = LLMInvocation(request_model=request_model)
    invocation.provider = "dashscope"
    invocation.operation_name = "generate_content"

    # Extract prompt as input message
    prompt = kwargs.get("prompt")
    if prompt:
        if isinstance(prompt, str):
            invocation.input_messages = [
                InputMessage(
                    role="user",
                    parts=[Text(content=prompt, type="text")],
                )
            ]
        elif isinstance(prompt, list):
            parts = []
            for p in prompt:
                if isinstance(p, str):
                    parts.append(Text(content=p, type="text"))
            if parts:
                invocation.input_messages = [
                    InputMessage(role="user", parts=parts)
                ]

    # Extract image URL if present (for image-to-video)
    img_url = kwargs.get("img_url")
    if img_url:
        if not invocation.input_messages:
            invocation.input_messages = []
        # Add image URL as a separate message or part
        if invocation.input_messages:
            invocation.input_messages[0].parts.append(
                Uri(uri=img_url, modality="image", mime_type=None, type="uri")
            )
        else:
            invocation.input_messages.append(
                InputMessage(
                    role="user",
                    parts=[
                        Uri(
                            uri=img_url,
                            modality="image",
                            mime_type=None,
                            type="uri",
                        )
                    ],
                )
            )

    return invocation


def _update_invocation_from_video_synthesis_response(
    invocation: LLMInvocation, response: Any
) -> None:
    """Update LLMInvocation with VideoSynthesis response data.

    Args:
        invocation: LLMInvocation to update
        response: VideoSynthesisResponse object
    """
    if not response:
        return

    try:
        # Extract token usage
        # FIXME: Usage of video synthesis is not expressed with input_tokens and output_tokens.
        input_tokens, output_tokens = _extract_usage(response)
        invocation.input_tokens = input_tokens
        invocation.output_tokens = output_tokens

        # Extract response model name (if available)
        try:
            response_model = getattr(response, "model", None)
            if response_model:
                invocation.response_model_name = response_model
        except (KeyError, AttributeError) as e:
            logger.debug(
                "Failed to extract response model from VideoSynthesis response: %s",
                e,
            )

        # Extract task_id from output
        output = getattr(response, "output", None)
        if output:
            task_id = None
            if hasattr(output, "get"):
                task_id = output.get("task_id")
            elif hasattr(output, "task_id"):
                task_id = getattr(output, "task_id", None)

            if task_id:
                invocation.response_id = task_id

            # Extract video URL from results
            video_url = None
            if hasattr(output, "get"):
                video_url = output.get("video_url")
            elif hasattr(output, "video_url"):
                video_url = getattr(output, "video_url", None)

            if video_url:
                invocation.output_messages = [
                    OutputMessage(
                        role="assistant",
                        parts=[
                            Uri(
                                uri=video_url,
                                modality="video",
                                mime_type=None,
                                type="uri",
                            )
                        ],
                        finish_reason="stop",
                    )
                ]

    except (KeyError, AttributeError) as e:
        logger.debug(
            "Failed to extract video URL from VideoSynthesis response: %s", e
        )


def _update_invocation_from_video_synthesis_async_response(
    invocation: LLMInvocation, response: Any
) -> None:
    """Update LLMInvocation with VideoSynthesis async_call response data.

    Args:
        invocation: LLMInvocation to update
        response: VideoSynthesisResponse object from async_call()
    """
    if not response:
        return

    try:
        output = getattr(response, "output", None)
        if output:
            task_id = None
            if hasattr(output, "get"):
                task_id = output.get("task_id")
            elif hasattr(output, "task_id"):
                task_id = getattr(output, "task_id", None)

            if task_id:
                invocation.response_id = task_id
    except (KeyError, AttributeError) as e:
        logger.debug(
            "Failed to extract task_id from VideoSynthesis async_call response: %s",
            e,
        )


# ============================================================================
# SpeechSynthesizer utilities
# ============================================================================


def _create_invocation_from_speech_synthesis(
    kwargs: dict, model: Optional[str] = None
) -> LLMInvocation:
    """Create LLMInvocation from SpeechSynthesizer.call kwargs.

    Args:
        kwargs: SpeechSynthesizer.call kwargs
        model: Model name (if not in kwargs)

    Returns:
        LLMInvocation object
    """
    request_model = kwargs.get("model") or model
    if not request_model:
        raise ValueError("Model name is required")

    invocation = LLMInvocation(request_model=request_model)
    invocation.provider = "dashscope"
    invocation.operation_name = "generate_content"

    # Extract text as input message
    text = kwargs.get("text")
    if text:
        invocation.input_messages = [
            InputMessage(
                role="user",
                parts=[Text(content=text, type="text")],
            )
        ]

    return invocation


def _update_invocation_from_speech_synthesis_response(
    invocation: LLMInvocation, response: Any, mime_type: Optional[str] = None
) -> None:
    """Update LLMInvocation with SpeechSynthesizer response data.

    Args:
        invocation: LLMInvocation to update
        response: SpeechSynthesisResult object
        mime_type: MIME type of audio (optional)
    """
    if not response:
        return

    try:
        # Extract request ID
        request_id = getattr(response, "request_id", None)
        if request_id:
            invocation.response_id = request_id

        # For TTS, the output is audio data (bytes)
        audio_data = getattr(response, "get_audio_data", None)
        if callable(audio_data):
            audio_bytes = audio_data()
            if audio_bytes:
                invocation.output_messages = [
                    OutputMessage(
                        role="assistant",
                        parts=[
                            Blob(
                                mime_type=mime_type,
                                modality="audio",
                                content=audio_bytes,
                            )
                        ],
                        finish_reason="stop",
                    )
                ]

    except (KeyError, AttributeError) as e:
        logger.debug(
            "Failed to update invocation from SpeechSynthesizer response: %s",
            e,
        )


def _create_invocation_from_speech_synthesis_v2(
    model: str, text: str
) -> LLMInvocation:
    """Create LLMInvocation from SpeechSynthesizerV2.call args.

    Args:
        model: Model name
        text: Text to synthesize

    Returns:
        LLMInvocation object
    """
    invocation = LLMInvocation(request_model=model)
    invocation.provider = "dashscope"
    invocation.operation_name = "generate_content"

    # Extract text as input message
    if text:
        invocation.input_messages = [
            InputMessage(
                role="user",
                parts=[Text(content=text, type="text")],
            )
        ]

    return invocation


def _update_invocation_from_speech_synthesis_v2_response(
    invocation: LLMInvocation,
    audio_data: bytes,
    mime_type: Optional[str] = None,
) -> None:
    """Update LLMInvocation with SpeechSynthesizerV2 response data.

    Args:
        invocation: LLMInvocation to update
        audio_data: Audio data bytes
        mime_type: MIME type of audio (optional)
    """
    if audio_data:
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[
                    Blob(
                        mime_type=mime_type,
                        modality="audio",
                        content=audio_data,
                    )
                ],
                finish_reason="stop",
            )
        ]


def _convert_speech_format_to_mime_type(speech_format: str) -> Optional[str]:
    """Convert from speech format to mime type.

    Args:
        speech_format: speech format of DashScope

    Returns:
        the mime type of speech
    """
    if speech_format == "wav":
        return "audio/wav"
    elif speech_format == "mp3":
        return "audio/mpeg"
    elif speech_format == "pcm":
        return "audio/pcm"
    elif speech_format == "opus":
        return "audio/opus"
    else:
        return None
