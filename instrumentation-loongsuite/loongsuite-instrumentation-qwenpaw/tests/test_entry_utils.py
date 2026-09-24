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

"""Unit tests for Entry argument parsing (no runtime package required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.qwenpaw._entry_utils import (
    build_entry_invocation,
    input_messages_from_msgs,
    parse_query_handler_call,
)


def test_parse_query_handler_call_positional():
    msgs = [1]
    req = SimpleNamespace(session_id="a")
    m, r = parse_query_handler_call((msgs, req), {})
    assert m is msgs
    assert r is req


def test_parse_query_handler_call_kwargs():
    msgs = [2]
    req = SimpleNamespace(session_id="b")
    m, r = parse_query_handler_call(tuple(), {"msgs": msgs, "request": req})
    assert m is msgs
    assert r is req


def test_build_entry_invocation_custom_attributes():
    inst = SimpleNamespace(agent_id="aid")
    req = SimpleNamespace(session_id="s", user_id="u", channel="feishu")
    inv = build_entry_invocation(inst, [], req)
    assert inv.session_id == "s"
    assert inv.user_id == "u"
    assert inv.attributes["qwenpaw.agent_id"] == "aid"
    assert inv.attributes["qwenpaw.channel"] == "feishu"
    assert inv.attributes["copaw.agent_id"] == "aid"
    assert inv.attributes["copaw.channel"] == "feishu"


def test_input_messages_from_agentscope_msg():
    pytest.importorskip("agentscope.message")
    from agentscope.message import Msg, TextBlock  # noqa: PLC0415

    m = Msg(name="u", role="user", content=[TextBlock(type="text", text="hi")])
    ims = input_messages_from_msgs([m])
    assert len(ims) == 1
    assert ims[0].role == "user"
    assert len(ims[0].parts) == 1
    assert ims[0].parts[0].content == "hi"
