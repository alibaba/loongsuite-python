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

"""Synthetic OpenAI JSON/SSE for the stable framework's text ReAct protocol."""

import json
import re

import httpx

from .runtime_provider import MODEL, USAGE, AsyncBytes, answer


class Provider:
    def __init__(self):
        self.requests = []
        self.responses = []

    def respond(self, request):
        body = json.loads(request.content)
        text = body["messages"][-1]["content"]
        current = text.rsplit("Question: ", 1)[-1]
        match = re.match(r"worker=([A-Za-z0-9_-]+) turn=(\d+)", current)
        assert match is not None, current
        worker, turn = match.group(1), int(match.group(2))
        index = current.count("Observation: ")
        assert index <= 3
        assert body["model"] == MODEL
        assert body["temperature"] == 0
        effort = body.get("reasoning_effort")
        assert effort in ("high", "none")
        if turn == 2:
            assert answer(worker, 1) in text
        tools = ["stock_lookup", "unit_price", "multiply"]
        arguments = dict(worker=worker, turn=turn)
        if index < 3:
            if index < 2:
                arguments["sku"] = "SKU-A"
            else:
                arguments.update(a=2, b=12)
            content = (
                "Thought: use the next inventory tool.\nAction: "
                + tools[index]
                + "\nAction Input: "
                + json.dumps(arguments)
            )
        else:
            assert "Observation: 7" in current
            assert "Observation: 12" in current
            assert "Observation: 24" in current
            content = "Thought: inventory checked.\nFinal Answer: " + answer(
                worker, turn
            )
        reasoning = (
            f"Synthetic public reasoning for {worker}/{turn}/{index}."
            if effort == "high"
            else ""
        )
        usage = json.loads(json.dumps(USAGE))
        if not reasoning:
            usage["completion_tokens_details"]["reasoning_tokens"] = 0
        identifier = f"synthetic-{worker}-t{turn}-r{index}"
        self.requests.append(body)
        self.responses.append(
            dict(
                worker=worker,
                turn=turn,
                round=index,
                id=identifier,
                content=content,
                reasoning=reasoning,
                usage=usage,
                tool=tools[index] if index < 3 else None,
                arguments=arguments if index < 3 else None,
            )
        )
        base = dict(id=identifier, model=MODEL, created=1)
        if not body["stream"]:
            return httpx.Response(
                200,
                json=dict(
                    base,
                    object="chat.completion",
                    choices=[
                        dict(
                            index=0,
                            message=dict(
                                role="assistant",
                                content=content,
                                reasoning_content=reasoning,
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=usage,
                ),
            )
        deltas = [dict(role="assistant", content="")]
        if reasoning:
            deltas += [dict(reasoning_content=reasoning[:15])]
            deltas += [dict(reasoning_content=reasoning[15:])]
        split = len(content) // 2
        deltas += [
            dict(content=content[:split]),
            dict(content=content[split:]),
        ]
        chunks = [
            dict(
                base,
                object="chat.completion.chunk",
                choices=[dict(index=0, delta=delta, finish_reason=None)],
            )
            for delta in deltas
        ]
        chunks.append(
            dict(
                base,
                object="chat.completion.chunk",
                choices=[dict(index=0, delta={}, finish_reason="stop")],
            )
        )
        chunks.append(
            dict(base, object="chat.completion.chunk", choices=[], usage=usage)
        )
        encoded = [
            ("data: " + json.dumps(chunk) + "\n\n").encode()
            for chunk in chunks
        ] + [b"data: [DONE]\n\n"]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"".join(encoded),
        )

    async def arespond(self, request):
        response = self.respond(request)
        if response.headers.get("content-type") == "text/event-stream":
            return httpx.Response(
                200,
                headers=response.headers,
                stream=AsyncBytes(
                    [
                        part + b"\n\n"
                        for part in response.content.split(b"\n\n")
                        if part
                    ]
                ),
            )
        return response

    def verify(self):
        for worker in ("auSync", "auStreamA", "auStreamB"):
            for turn in (1, 2):
                rounds = [
                    x["round"]
                    for x in self.responses
                    if x["worker"] == worker and x["turn"] == turn
                ]
                assert rounds == [0, 1, 2, 3]
        assert len(self.requests) == 24
        return dict(request_count=24, provider_total_tokens=24 * 30)
