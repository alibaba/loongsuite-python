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

"""Unit tests for opentelemetry.instrumentation.openhands.internal.utils."""

from __future__ import annotations

import json

from opentelemetry.instrumentation.openhands.internal.utils import (
    _to_jsonable,
    action_to_genai_output,
    extract_uuid_str,
    maybe_preview,
    maybe_to_json_str,
    messages_to_genai_input,
    preview,
    safe_get_attr,
    safe_str,
    serialize_message,
    to_json_str,
)

# ---------------------------------------------------------------------------
# safe_str
# ---------------------------------------------------------------------------


def test_safe_str_none():
    assert safe_str(None) == ""


def test_safe_str_normal():
    assert safe_str("hello") == "hello"
    assert safe_str(42) == "42"


def test_safe_str_exception():
    class Bad:
        def __str__(self):
            raise RuntimeError("bad")

    assert safe_str(Bad()) == ""


# ---------------------------------------------------------------------------
# preview / maybe_preview
# ---------------------------------------------------------------------------


def test_preview_returns_full_text():
    text = "hello world"
    assert preview(text) == "hello world"
    assert preview(text, max_len=5) == "hello world"


def test_maybe_preview_alias():
    assert maybe_preview("abc") == "abc"


def test_preview_none():
    assert preview(None) == ""


# ---------------------------------------------------------------------------
# safe_get_attr
# ---------------------------------------------------------------------------


def test_safe_get_attr_single():
    class Obj:
        x = 42

    assert safe_get_attr(Obj(), "x") == 42


def test_safe_get_attr_multiple_names():
    class Obj:
        b = "found"

    assert safe_get_attr(Obj(), "a", "b") == "found"


def test_safe_get_attr_none_obj():
    assert safe_get_attr(None, "x") is None
    assert safe_get_attr(None, "x", default="d") == "d"


def test_safe_get_attr_missing():
    class Obj:
        pass

    assert safe_get_attr(Obj(), "x") is None
    assert safe_get_attr(Obj(), "x", default="fallback") == "fallback"


def test_safe_get_attr_raises():
    class Obj:
        @property
        def x(self):
            raise RuntimeError("boom")

    assert safe_get_attr(Obj(), "x") is None


# ---------------------------------------------------------------------------
# serialize_message
# ---------------------------------------------------------------------------


def test_serialize_message_none():
    assert serialize_message(None) == ""


def test_serialize_message_str():
    assert serialize_message("hello") == "hello"


def test_serialize_message_with_text_attr():
    class Msg:
        text = "the text"

    assert serialize_message(Msg()) == "the text"


def test_serialize_message_with_content_attr():
    class Msg:
        content = "the content"

    assert serialize_message(Msg()) == "the content"


def test_serialize_message_with_value_attr():
    class Msg:
        value = "the value"

    assert serialize_message(Msg()) == "the value"


def test_serialize_message_with_list_content():
    class Part:
        text = "part1"

    class Msg:
        text = None
        content = [Part()]
        value = None

    assert "part1" in serialize_message(Msg())


def test_serialize_message_fallback():
    class Msg:
        def __str__(self):
            return "fallback"

    assert serialize_message(Msg()) == "fallback"


# ---------------------------------------------------------------------------
# extract_uuid_str
# ---------------------------------------------------------------------------


def test_extract_uuid_str_none():
    assert extract_uuid_str(None) == ""


def test_extract_uuid_str_with_hex():
    class UUID:
        hex = "abcdef1234567890"

    assert extract_uuid_str(UUID()) == "abcdef1234567890"


def test_extract_uuid_str_string():
    assert extract_uuid_str("some-uuid") == "some-uuid"


def test_extract_uuid_str_no_hex():
    """When hex attr is None or missing, falls back to str()."""
    assert extract_uuid_str(42) == "42"


# ---------------------------------------------------------------------------
# _to_jsonable
# ---------------------------------------------------------------------------


def test_to_jsonable_primitives():
    assert _to_jsonable(None) is None
    assert _to_jsonable(True) is True
    assert _to_jsonable(42) == 42
    assert _to_jsonable(3.14) == 3.14
    assert _to_jsonable("hello") == "hello"


def test_to_jsonable_dict():
    assert _to_jsonable({"a": 1}) == {"a": 1}


def test_to_jsonable_list():
    assert _to_jsonable([1, 2, 3]) == [1, 2, 3]


def test_to_jsonable_tuple():
    assert _to_jsonable((1, 2)) == [1, 2]


def test_to_jsonable_set():
    result = _to_jsonable({1})
    assert isinstance(result, list)
    assert 1 in result


def test_to_jsonable_max_depth():
    nested = {"a": {"b": {"c": 1}}}
    result = _to_jsonable(nested, depth=0, max_depth=1)
    # At max_depth, dicts become safe_str
    assert isinstance(result, dict)
    assert isinstance(result["a"], str)


def test_to_jsonable_pydantic_model_dump():
    class PydanticLike:
        def model_dump(self):
            return {"field": "value"}

    result = _to_jsonable(PydanticLike())
    assert result == {"field": "value"}


def test_to_jsonable_pydantic_model_dump_error():
    class BadPydantic:
        def model_dump(self):
            raise RuntimeError("fail")

        def __str__(self):
            return "fallback"

    result = _to_jsonable(BadPydantic())
    assert result == "fallback"


def test_to_jsonable_object_with_dict():
    class Obj:
        def __init__(self):
            self.x = 1
            self.y = "two"
            self._private = "skip"

    result = _to_jsonable(Obj())
    assert isinstance(result, dict)
    assert result["x"] == 1
    assert result["y"] == "two"
    assert "_private" not in result


def test_to_jsonable_object_with_empty_dict():
    class Obj:
        def __init__(self):
            self._only_private = True

    result = _to_jsonable(Obj())
    # Only private attrs → empty dict, so falls through to safe_str
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# to_json_str / maybe_to_json_str
# ---------------------------------------------------------------------------


def test_to_json_str_basic():
    assert to_json_str({"a": 1}) == '{"a": 1}'


def test_to_json_str_empty():
    assert to_json_str("") == '""'


def test_to_json_str_none():
    result = to_json_str(None)
    assert result == "null"


def test_to_json_str_exception():
    class NotSerializable:
        def __str__(self):
            return "fallback"

        def __repr__(self):
            raise RuntimeError("boom")

    # Even if _to_jsonable succeeds, json.dumps might fail;
    # then we fall back to safe_str.
    result = to_json_str(NotSerializable())
    assert isinstance(result, str)


def test_maybe_to_json_str_alias():
    assert maybe_to_json_str({"key": "val"}) == to_json_str({"key": "val"})


# ---------------------------------------------------------------------------
# messages_to_genai_input
# ---------------------------------------------------------------------------


def test_messages_to_genai_input_not_list():
    assert messages_to_genai_input("not a list") == ""
    assert messages_to_genai_input(None) == ""


def test_messages_to_genai_input_basic():
    msgs = [{"role": "user", "content": "hello"}]
    result = messages_to_genai_input(msgs)
    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["role"] == "user"
    assert parsed[0]["content"] == "hello"


def test_messages_to_genai_input_with_attr_objects():
    class Msg:
        role = "assistant"
        content = "hi there"

    result = messages_to_genai_input([Msg()])
    parsed = json.loads(result)
    assert parsed[0]["role"] == "assistant"
    assert parsed[0]["content"] == "hi there"


def test_messages_to_genai_input_with_list_content():
    class Part:
        text = "part1"
        content = None

    msg = {"role": "user", "content": [Part()]}
    result = messages_to_genai_input([msg])
    parsed = json.loads(result)
    assert "part1" in parsed[0]["content"]


def test_messages_to_genai_input_with_tool_calls():
    class Msg:
        role = "assistant"
        content = "thinking"
        tool_calls = [{"id": "tc1", "function": {"name": "bash"}}]

    result = messages_to_genai_input([Msg()])
    parsed = json.loads(result)
    assert "tool_calls" in parsed[0]


# ---------------------------------------------------------------------------
# action_to_genai_output
# ---------------------------------------------------------------------------


def test_action_to_genai_output_none():
    assert action_to_genai_output(None) == ""


def test_action_to_genai_output_basic():
    class Action:
        action = "run"
        thought = "thinking"
        command = "ls"
        code = None
        path = None
        url = None
        content = None
        task_list = None
        name = None
        arguments = None

    result = action_to_genai_output(Action())
    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["role"] == "assistant"
    assert parsed[0]["content"] == "thinking"
    assert parsed[0]["tool_calls"][0]["function"]["name"] == "run"
    assert (
        parsed[0]["tool_calls"][0]["function"]["arguments"]["command"] == "ls"
    )


def test_action_to_genai_output_no_thought():
    class Action:
        action = "finish"
        thought = ""
        command = None
        code = None
        path = None
        url = None
        content = None
        task_list = None
        name = None
        arguments = None

    result = action_to_genai_output(Action())
    parsed = json.loads(result)
    assert "content" not in parsed[0]
    assert parsed[0]["tool_calls"][0]["function"]["name"] == "finish"
