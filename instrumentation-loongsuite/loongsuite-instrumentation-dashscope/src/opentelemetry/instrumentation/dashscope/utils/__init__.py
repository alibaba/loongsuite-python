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

"""Utility functions for DashScope instrumentation.

This module re-exports all utility functions from submodules for convenient imports.
"""

from __future__ import annotations

# Common utilities
from .common import (
    _SKIP_INSTRUMENTATION_KEY,
    DASHSCOPE_SERVER_ADDRESS,
    DASHSCOPE_SERVER_PORT,
    _extract_task_id,
    _extract_usage,
    _get_parameter,
)

# Generation utilities
from .generation import (
    _create_accumulated_response,
    _create_invocation_from_generation,
    _extract_input_messages,
    _extract_output_messages,
    _extract_tool_definitions,
    _update_invocation_from_response,
)

# Multimodal utilities
from .multimodal import (
    # SpeechSynthesis
    _convert_speech_format_to_mime_type,
    # ImageSynthesis
    _create_invocation_from_image_synthesis,
    # MultiModalConversation
    _create_invocation_from_multimodal_conversation,
    _create_invocation_from_speech_synthesis,
    _create_invocation_from_speech_synthesis_v2,
    # VideoSynthesis
    _create_invocation_from_video_synthesis,
    _extract_multimodal_input_messages,
    _extract_multimodal_output_messages,
    _update_invocation_from_image_synthesis_async_response,
    _update_invocation_from_image_synthesis_response,
    _update_invocation_from_multimodal_response,
    _update_invocation_from_speech_synthesis_response,
    _update_invocation_from_speech_synthesis_v2_response,
    _update_invocation_from_video_synthesis_async_response,
    _update_invocation_from_video_synthesis_response,
)

__all__ = [
    # Common
    "_get_parameter",
    "_extract_usage",
    "_extract_task_id",
    "_SKIP_INSTRUMENTATION_KEY",
    "DASHSCOPE_SERVER_ADDRESS",
    "DASHSCOPE_SERVER_PORT",
    # Generation
    "_extract_input_messages",
    "_extract_tool_definitions",
    "_extract_output_messages",
    "_create_invocation_from_generation",
    "_update_invocation_from_response",
    "_create_accumulated_response",
    # ImageSynthesis
    "_create_invocation_from_image_synthesis",
    "_update_invocation_from_image_synthesis_response",
    "_update_invocation_from_image_synthesis_async_response",
    # MultiModalConversation
    "_extract_multimodal_input_messages",
    "_extract_multimodal_output_messages",
    "_create_invocation_from_multimodal_conversation",
    "_update_invocation_from_multimodal_response",
    # VideoSynthesis
    "_create_invocation_from_video_synthesis",
    "_update_invocation_from_video_synthesis_response",
    "_update_invocation_from_video_synthesis_async_response",
    # SpeechSynthesis
    "_convert_speech_format_to_mime_type",
    "_create_invocation_from_speech_synthesis",
    "_update_invocation_from_speech_synthesis_response",
    "_create_invocation_from_speech_synthesis_v2",
    "_update_invocation_from_speech_synthesis_v2_response",
]
