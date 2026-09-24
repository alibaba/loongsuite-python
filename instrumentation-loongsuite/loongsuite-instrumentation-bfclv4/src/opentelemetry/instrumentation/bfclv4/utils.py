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

"""Helpers for the BFCL v4 instrumentation.

The :class:`GenAIHookHelper` mirrors the helper used by the LoongSuite CrewAI
instrumentation: it gates ``gen_ai.input.messages`` /
``gen_ai.output.messages`` / ``gen_ai.system_instructions`` on the standard
LoongSuite content-capture environment knobs so that prompt content is not
exported by default.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import Span
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
)
from opentelemetry.util.genai.utils import (
    gen_ai_json_dumps,
    get_content_capturing_mode,
    is_experimental_mode,
)

logger = logging.getLogger(__name__)


class GenAIHookHelper:
    """Conditionally write prompt / completion content to the span."""

    def __init__(self, capture_content: bool = True) -> None:
        self.capture_content = capture_content

    def on_completion(
        self,
        span: Span,
        inputs: Optional[List[InputMessage]] = None,
        outputs: Optional[List[OutputMessage]] = None,
        system_instructions: Optional[List[MessagePart]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not span.is_recording():
            return

        if self.capture_content and is_experimental_mode():
            mode = get_content_capturing_mode()
            should_capture_span = mode in (
                ContentCapturingMode.SPAN_ONLY,
                ContentCapturingMode.SPAN_AND_EVENT,
            )

            if should_capture_span:
                if inputs:
                    span.set_attribute(
                        gen_ai_attributes.GEN_AI_INPUT_MESSAGES,
                        gen_ai_json_dumps(
                            [dataclasses.asdict(i) for i in inputs]
                        ),
                    )
                if outputs:
                    span.set_attribute(
                        gen_ai_attributes.GEN_AI_OUTPUT_MESSAGES,
                        gen_ai_json_dumps(
                            [dataclasses.asdict(o) for o in outputs]
                        ),
                    )
                if system_instructions:
                    span.set_attribute(
                        gen_ai_attributes.GEN_AI_SYSTEM_INSTRUCTIONS,
                        gen_ai_json_dumps(
                            [
                                dataclasses.asdict(s)
                                for s in system_instructions
                            ]
                        ),
                    )

        if attributes:
            for key, value in attributes.items():
                if value is None:
                    continue
                try:
                    span.set_attribute(key, value)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "bfclv4: failed to set attribute %s",
                        key,
                        exc_info=True,
                    )


def to_text_input(role: str, content: Any) -> List[InputMessage]:
    if content in (None, "", [], {}):
        return []
    text = content if isinstance(content, str) else _to_safe_str(content)
    return [InputMessage(role=role, parts=[Text(content=text)])]


def to_text_output(
    role: str, content: Any, finish_reason: str = "stop"
) -> List[OutputMessage]:
    if content in (None, "", [], {}):
        return []
    text = content if isinstance(content, str) else _to_safe_str(content)
    return [
        OutputMessage(
            role=role, parts=[Text(content=text)], finish_reason=finish_reason
        )
    ]


def _to_safe_str(value: Any) -> str:
    """Best-effort JSON serialisation, falling back to ``str()``.

    The wrapper code never wants a serialisation failure to break a span.
    """
    try:
        return gen_ai_json_dumps(value)
    except Exception:  # noqa: BLE001
        try:
            return str(value)
        except Exception:  # noqa: BLE001
            return "<unserialisable>"


def truncate_text(value: str, limit: int = 4096) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"...<truncated {len(value) - limit} chars>"
