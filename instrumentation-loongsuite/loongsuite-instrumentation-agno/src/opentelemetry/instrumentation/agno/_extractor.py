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

import json
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Tuple,
)

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.types import AttributeValue


class AgentRunRequestExtractor(object):
    def extract(
        self, agent: Any, arguments: Dict[Any, Any]
    ) -> Iterable[Tuple[str, AttributeValue]]:
        if agent.name:
            yield GenAIAttributes.GEN_AI_AGENT_NAME, f"{agent.name}"

        if agent.session_id:
            yield GenAIAttributes.GEN_AI_AGENT_ID, f"{agent.session_id}"

        if agent.knowledge:
            yield (
                f"{GenAIAttributes.GEN_AI_AGENT_NAME}.knowledge",
                f"{agent.knowledge.__class__.__name__}",
            )

        if agent.tools:
            tool_names = []
            from agno.tools.function import Function  # noqa: PLC0415
            from agno.tools.toolkit import Toolkit  # noqa: PLC0415

            for tool in agent.tools:
                if isinstance(tool, Function):
                    tool_names.append(tool.name)
                elif isinstance(tool, Toolkit):
                    tool_names.extend([f for f in tool.functions.keys()])
                elif callable(tool):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append(str(tool))
            yield GenAIAttributes.GEN_AI_TOOL_NAME, ", ".join(tool_names)

        for item in arguments.items():
            key, entry_value = item
            if key == "run_response":
                yield (
                    GenAIAttributes.GEN_AI_RESPONSE_ID,
                    f"{entry_value.run_id}",
                )
            elif key == "run_messages":
                messages = entry_value.messages
                for idx in range(len(messages)):
                    message = messages[idx]
                    yield (
                        f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.message",
                        f"{json.dumps(message.to_dict(), indent=2)}",
                    )
            elif key == "response_format":
                yield (
                    GenAIAttributes.GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT,
                    f"{entry_value}",
                )


class AgentRunResponseExtractor(object):
    def extract(self, response: Any) -> Iterable[Tuple[str, AttributeValue]]:
        yield (
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS,
            f"{response.to_json()}",
        )


class FunctionCallRequestExtractor(object):
    def extract(
        self, function_call: Any
    ) -> Iterable[Tuple[str, AttributeValue]]:
        if function_call.function.name:
            yield (
                GenAIAttributes.GEN_AI_TOOL_NAME,
                f"{function_call.function.name}",
            )

        if function_call.function.description:
            yield (
                GenAIAttributes.GEN_AI_TOOL_DESCRIPTION,
                f"{function_call.function.description}",
            )

        if function_call.call_id:
            yield (
                GenAIAttributes.GEN_AI_TOOL_CALL_ID,
                f"{function_call.call_id}",
            )

        if function_call.arguments:
            yield (
                f"{GenAIAttributes.GEN_AI_TOOL_TYPE}.arguments",
                f"{json.dumps(function_call.arguments, indent=2)}",
            )


class FunctionCallResponseExtractor(object):
    def extract(self, response: Any) -> Iterable[Tuple[str, AttributeValue]]:
        yield (
            f"{GenAIAttributes.GEN_AI_TOOL_TYPE}.response",
            f"{response.result}",
        )


class ModelRequestExtractor(object):
    def extract(
        self, model: Any, arguments: Dict[Any, Any]
    ) -> Iterable[Tuple[str, AttributeValue]]:
        request_kwargs = {}
        if getattr(model, "request_kwargs", None):
            request_kwargs = model.request_kwargs
        if getattr(model, "request_params", None):
            request_kwargs = model.request_params
        if getattr(model, "get_request_kwargs", None):
            request_kwargs = model.get_request_kwargs()
        if getattr(model, "get_request_params", None):
            request_kwargs = model.get_request_params()

        if request_kwargs:
            yield (
                GenAIAttributes.GEN_AI_REQUEST_MODEL,
                f"{json.dumps(request_kwargs, indent=2)}",
            )

        for item in arguments.items():
            key, entry_value = item
            if key == "response_format":
                yield (
                    GenAIAttributes.GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT,
                    f"{entry_value}",
                )
            elif key == "messages":
                messages = entry_value
                for idx in range(len(messages)):
                    message = messages[idx]
                    yield (
                        f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.message",
                        f"{json.dumps(message.to_dict(), indent=2)}",
                    )
            elif key == "tools":
                tools = entry_value
                for idx in range(len(tools)):
                    yield (
                        f"{GenAIAttributes.GEN_AI_TOOL_DESCRIPTION}.{idx}",
                        f"{json.dumps(tools[idx], indent=2)}",
                    )


class ModelResponseExtractor(object):
    def extract(
        self, responses: List[Any]
    ) -> Iterable[Tuple[str, AttributeValue]]:
        content = ""
        for response in responses:
            # basic response fields
            if getattr(response, "role", None):
                yield (
                    f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.role",
                    response.role,
                )
            if getattr(response, "content", None):
                content += response.content
            if getattr(response, "audio", None):
                yield (
                    f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.audio",
                    json.dumps(response.audio.to_dict(), indent=2),
                )
            if getattr(response, "image", None):
                yield (
                    f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.image",
                    json.dumps(response.image.to_dict(), indent=2),
                )
            for idx, tool_execution in enumerate(
                getattr(response, "tool_executions", []) or []
            ):
                yield (
                    f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.tool_executions.{idx}",
                    json.dumps(tool_execution.to_dict(), indent=2),
                )
            # other metadata
            if getattr(response, "event", None):
                yield (
                    f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.event",
                    response.event,
                )
            if getattr(response, "provider_data", None):
                yield (
                    f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.provider_data",
                    json.dumps(response.provider_data, indent=2),
                )
            if getattr(response, "thinking", None):
                yield (
                    f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.thinking",
                    response.thinking,
                )
            if getattr(response, "redacted_thinking", None):
                yield (
                    f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.redacted_thinking",
                    response.redacted_thinking,
                )
            if getattr(response, "reasoning_content", None):
                yield (
                    f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.reasoning_content",
                    response.reasoning_content,
                )
            if getattr(response, "extra", None):
                yield (
                    f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.extra",
                    json.dumps(response.extra, indent=2),
                )
        if len(content):
            yield (
                f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.content",
                f"{content}",
            )
