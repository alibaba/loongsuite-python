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

"""Stable native ReAct/AgentExecutor; only the HTTP provider is scripted."""

import asyncio
import threading
import time
from collections import Counter
from queue import Queue
from typing import Any, ClassVar

import httpx
from agentuniverse.agent.action.tool.tool import Tool
from agentuniverse.agent.action.tool.tool_manager import ToolManager
from agentuniverse.agent.agent_model import AgentModel
from agentuniverse.agent.template.react_agent_template import (
    ReActAgentTemplate,
)
from agentuniverse.llm.llm import LLM
from agentuniverse.llm.llm_channel.llm_channel import LLMChannel
from agentuniverse.llm.llm_manager import LLMManager
from agentuniverse.prompt.prompt import Prompt
from openai import AsyncOpenAI, OpenAI

from opentelemetry import context

import importlib.metadata as _metadata

def _framework_version():
    raw = _metadata.version("agentUniverse")
    parts = raw.split(".")
    return (int(parts[0]), int(parts[1]))

IS_ALPHA = _framework_version() >= (0, 1)

from .runtime_provider import MODEL, Provider as StructuredProvider, answer, prompt
if IS_ALPHA:
    Provider = StructuredProvider
else:
    from .stable_runtime_provider import Provider


class ChannelLLM(LLM):
    def _call(self, *args, **kwargs):
        raise AssertionError("The real LLM channel should handle the call")

    async def _acall(self, *args, **kwargs):
        raise AssertionError("The real LLM channel should handle the call")

    def get_num_tokens(self, text):
        return len(text.split())


class InventoryTool(Tool):
    workload: ClassVar[Any] = None

    def execute(self, **kwargs):
        work = self.workload
        with work.lock:
            work.active += 1
            work.max_active = max(work.max_active, work.active)
        try:
            time.sleep(0.008)
            value = (
                7
                if self.name == "stock_lookup"
                else 12
                if self.name == "unit_price"
                else kwargs["a"] * kwargs["b"]
            )
            work.tool_calls.append(
                {"name": self.name, "args": kwargs, "result": value}
            )
            return str(value)
        finally:
            with work.lock:
                work.active -= 1


class InventoryAgent(ReActAgentTemplate):
    """Configure resources; execute and tool dispatch remain framework-owned."""

    configured_llm: Any = None
    configured_prompt: Any = None

    def process_memory(self, agent_input, **kwargs):
        return None

    def process_llm(self, **kwargs):
        return self.configured_llm

    def process_prompt(self, agent_input, **kwargs):
        return self.configured_prompt


class Workload:
    def __init__(self, reasoning=True):
        InventoryTool.workload = self
        self.backend = Provider()
        self.reasoning = reasoning
        self.tool_calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.names = []
        for name, fields in (
            ("stock_lookup", ["sku"]),
            ("unit_price", ["sku"]),
            ("multiply", ["a", "b"]),
        ):
            tool = InventoryTool(
                name=name,
                description="Synthetic inventory tool; supply JSON arguments",
                input_keys=["worker", "turn"] + fields,
            )
            ToolManager().register(tool.get_instance_code(), tool)
            self.names.append(name)

    def agent(self, label, streaming):
        channel = LLMChannel(
            channel_name="openai",
            channel_model_name=MODEL,
            channel_api_key="synthetic",
            channel_api_base="http://provider.invalid/v1",
            ext_params={
                "reasoning_effort": "high" if self.reasoning else "none",
                "stream_options": {"include_usage": True},
            },
        )
        channel.channel_model_config = dict(
            model=MODEL,
            streaming=streaming,
            temperature=0,
            max_tokens=512,
            request_timeout=30,
            max_retries=0,
        )
        channel.client = OpenAI(
            api_key="synthetic",
            base_url="http://provider.invalid/v1",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(self.backend.respond)
            ),
        )
        channel.async_client = AsyncOpenAI(
            api_key="synthetic",
            base_url="http://provider.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(self.backend.arespond)
            ),
        )
        llm = ChannelLLM(
            name="inventory_llm", model_name=MODEL, streaming=streaming
        )
        llm._channel_instance = channel
        llm._max_context_length = 8192
        # The stable SDK's Agent.invoke_llm resolves the LLM by name through
        # LLMManager; a bare process_llm override is never consulted. Register
        # the channel-backed instance so judge_stream sees a real LLM.
        llm_code = llm.get_instance_code()
        LLMManager().register(llm_code, llm)
        self._llm_code = llm_code
        agent = InventoryAgent()
        profile = {
            "llm_model": dict(
                name="inventory_llm",
                model_name=MODEL,
                temperature=0,
                streaming=streaming,
            )
        }
        if IS_ALPHA:
            # 0.1.0a2 builds the user message from profile['instruction']
            # rendered with input_dict; without it messages come back empty.
            profile["instruction"] = "{input}"
        agent.agent_model = AgentModel(
            info={"name": "inventory_" + label},
            profile=profile,
        )
        agent.agent_model.action = {"tool": self.names}
        # SDK 0.0.19.1 InventoryAgent does not accept a max_iterations field;
        # leave the ReAct loop at its SDK default instead of setting a custom one.
        agent.configured_llm = llm
        agent.configured_prompt = Prompt(
            prompt_template=(
                "Use the inventory tools: {tools}\nNames: {tool_names}\n"
                "History: {chat_history}\nQuestion: {input}\n{agent_scratchpad}"
            ),
            input_variables=[
                "tools",
                "tool_names",
                "chat_history",
                "input",
                "agent_scratchpad",
            ],
        )
        return agent

    async def run(self, tracer_provider):
        records = []

        async def worker(label, streaming, synchronous=False):
            agent = self.agent(label, streaming)
            history = ""
            with tracer_provider.get_tracer("app").start_as_current_span(
                "au-session." + label
            ):
                before = context.get_current()
                for turn in (1, 2):
                    kwargs = dict(
                        input=prompt(label, turn),
                        session_id=label,
                        chat_history=history,
                        output_stream=Queue(),
                    )
                    output = (
                        agent.run(**kwargs)
                        if synchronous
                        else await agent.async_run(**kwargs)
                    )
                    text = output.get_data("output")
                    assert text == answer(label, turn)
                    assert context.get_current() is before
                    history += f"Human: {kwargs['input']}\nAI: {text}\n"
                    records.append(
                        dict(
                            worker=label,
                            turn=turn,
                            streaming=streaming,
                            text=text,
                        )
                    )
            # On alpha the registered LLM is shared via LLMManager and each
            # agent gets a copied channel; closing the shared original here
            # would poison later workers. MockTransport clients need no close.
            if not IS_ALPHA:
                agent.configured_llm._channel_instance.client.close()
                await agent.configured_llm._channel_instance.async_client.close()

        await worker("auSync", False, True)
        await asyncio.gather(
            worker("auStreamA", True), worker("auStreamB", True)
        )
        # Drop the registered channel-backed LLM so it cannot leak into other
        # tests; MockTransport clients hold no real sockets.
        code = getattr(self, "_llm_code", None)
        if code and code in LLMManager()._instance_obj_map:
            LLMManager().unregister(code)
        counts = Counter(
            (item["args"]["worker"], item["args"]["turn"], item["name"])
            for item in self.tool_calls
        )
        for label in ("auSync", "auStreamA", "auStreamB"):
            for turn in (1, 2):
                for name in ("stock_lookup", "unit_price", "multiply"):
                    assert counts[label, turn, name] == 1
        return dict(
            self.backend.verify(),
            scenarios=records,
            tool_calls=self.tool_calls,
            max_active_tools=self.max_active,
        )
