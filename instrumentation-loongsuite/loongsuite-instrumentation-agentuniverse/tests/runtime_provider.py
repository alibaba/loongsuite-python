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

"""Deterministic OpenAI HTTP/SSE boundary, not an Agent-loop replacement.

Actual OpenAI SDK builds requests/parses JSON or SSE; actual framework decides
when to invoke tools and calls the model again. All text is synthetic.
"""

import asyncio
import json
import re

import httpx

MODEL = "synthetic-reasoning-model"
USAGE = {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30,
    "prompt_tokens_details": {"cached_tokens": 5},
    "completion_tokens_details": {"reasoning_tokens": 3},
}


def prompt(worker, turn):
    return f"worker={worker} turn={turn} Check stock and unit price for SKU-A, then price two units."


def answer(worker, turn):
    return f"[{worker}] turn{turn}: stock=7 total=24"


def content_text(content):
    if isinstance(content, str):
        return content
    return "".join(
        p.get("text", "") for p in (content or []) if isinstance(p, dict)
    )


class AsyncBytes(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for x in self.chunks:
            await asyncio.sleep(0.002)
            yield x


class Provider:
    def __init__(self):
        self.requests = []
        self.responses = []

    @staticmethod
    def validate_body(body):
        messages = body["messages"]
        index = max(i for i, m in enumerate(messages) if m["role"] == "user")
        match = re.search(
            r"worker=([A-Za-z0-9_-]+) turn=(\d+)",
            content_text(messages[index].get("content")),
        )
        worker, turn = match.group(1), int(match.group(2))
        plans = {
            tc["id"]: tc
            for m in messages[index + 1 :]
            for tc in m.get("tool_calls", [])
        }
        for message in messages[index + 1 :]:
            if message["role"] != "tool":
                continue
            tc = plans[message["tool_call_id"]]
            args = json.loads(tc["function"]["arguments"])
            assert args["worker"] == worker and args["turn"] == turn
            expected = {
                "stock_lookup": "7",
                "unit_price": "12",
                "multiply": "24",
            }[tc["function"]["name"]]
            assert content_text(message.get("content")).strip() == expected, (
                message
            )
        if turn == 2:
            assert any(
                m["role"] == "assistant"
                and answer(worker, 1) in content_text(m.get("content"))
                for m in messages[:index]
            ), (worker, "prior conversation was not retained")
        assert body.get("reasoning_effort") == "high"

    def respond(self, req):
        body = json.loads(req.content)
        self.validate_body(body)
        messages = body["messages"]
        index = max(i for i, m in enumerate(messages) if m["role"] == "user")
        text = content_text(messages[index].get("content"))
        match = re.search(r"worker=([A-Za-z0-9_-]+) turn=(\d+)", text)
        assert match, text
        worker, turn = match.group(1), int(match.group(2))
        tail = messages[index + 1 :]
        tool_results = [m for m in tail if m["role"] == "tool"]
        n = len(tool_results)
        if n == 0:
            tools = [
                (
                    "stock_lookup",
                    {"worker": worker, "turn": turn, "sku": "SKU-A"},
                ),
                (
                    "unit_price",
                    {"worker": worker, "turn": turn, "sku": "SKU-A"},
                ),
            ]
            round_index = 0
        elif n == 2:
            tools = [
                ("multiply", {"worker": worker, "turn": turn, "a": 2, "b": 12})
            ]
            round_index = 1
        elif n == 3:
            tools = []
            round_index = 2
        else:
            raise AssertionError(f"Unexpected actual tool result count: {n}")
        ids = [
            f"{worker}-t{turn}-r{round_index}-{i}" for i in range(len(tools))
        ]
        calls = [
            {
                "id": ids[i],
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(tools)
        ]
        final_text = None if tools else answer(worker, turn)
        reasoning = f"synthetic-reasoning:{worker}:t{turn}:r{round_index}"
        message = {
            "role": "assistant",
            "content": final_text,
            "reasoning_content": reasoning,
        }
        if calls:
            message["tool_calls"] = calls
        finish = "tool_calls" if tools else "stop"
        self.requests.append(
            {
                "worker": worker,
                "turn": turn,
                "round": round_index,
                "request": body,
                "received_tool_ids": [
                    m.get("tool_call_id") for m in tool_results
                ],
            }
        )
        self.responses.append(
            {
                "worker": worker,
                "turn": turn,
                "round": round_index,
                "reasoning_content": reasoning,
                "tool_calls": calls,
                "text": final_text,
                "usage": USAGE,
            }
        )
        base = {
            "id": f"synthetic-{worker}-{turn}-{round_index}",
            "created": 1,
            "model": MODEL,
        }
        if not body.get("stream"):
            return httpx.Response(
                200,
                json={
                    **base,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": finish,
                        }
                    ],
                    "usage": USAGE,
                },
            )
        events = []

        def emit(delta, finish_reason=None):
            events.append(
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            )

        emit({"role": "assistant"})
        for part in [reasoning[:12], reasoning[12:]]:
            emit({"reasoning_content": part})
        for i, call in enumerate(calls):
            args = call["function"]["arguments"]
            emit(
                {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": "",
                            },
                        }
                    ]
                }
            )
            for part in [args[:9], args[9:]]:
                emit(
                    {
                        "tool_calls": [
                            {"index": i, "function": {"arguments": part}}
                        ]
                    }
                )
        if final_text:
            for part in [final_text[:12], final_text[12:]]:
                emit({"content": part})
        emit({}, finish)
        events.append(
            {
                **base,
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": USAGE,
            }
        )
        lines = [
            ("data: " + json.dumps(e) + "\n\n").encode() for e in events
        ] + [b"data: [DONE]\n\n"]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"".join(lines),
        )

    async def arespond(self, req):
        await asyncio.sleep(0.003)
        response = self.respond(req)
        if response.headers.get("content-type", "").startswith(
            "text/event-stream"
        ):
            return httpx.Response(
                200,
                headers=response.headers,
                stream=AsyncBytes(response.content.splitlines(keepends=True)),
            )
        return response

    def verify(self):
        for item in self.requests:
            self.validate_body(item["request"])
        groups = {}
        for item in self.requests:
            groups.setdefault((item["worker"], item["turn"]), []).append(
                item["round"]
            )
        for key, rounds in groups.items():
            assert rounds == [0, 1, 2], (key, rounds)
        return {
            "request_count": len(self.requests),
            "turn_count": len(groups),
            "reasoning_enabled": True,
            "reasoning_response_count": len(self.responses),
            "provider_usage_per_request": USAGE,
        }
