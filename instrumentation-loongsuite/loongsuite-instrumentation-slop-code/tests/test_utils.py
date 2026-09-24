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

"""Tests for utility functions."""

from unittest.mock import MagicMock

from opentelemetry.instrumentation.slop_code.utils import (
    MAX_ATTR_LEN,
    genai_messages,
    json_dumps_attr,
    safe_get,
    safe_get_nested,
    set_optional_attr,
    truncate_text,
)


class TestSafeGet:
    def test_safe_get_normal(self):
        obj = MagicMock()
        obj.foo = "bar"
        assert safe_get(obj, "foo") == "bar"

    def test_safe_get_default(self):
        obj = MagicMock(spec=[])
        assert safe_get(obj, "missing", "default") == "default"

    def test_safe_get_exception(self):
        """safe_get should return default when getattr raises."""

        class Broken:
            @property
            def bad(self):
                raise ValueError("broken property")

            def __getattr__(self, name):
                raise TypeError("broken getattr")

        obj = Broken()
        assert safe_get(obj, "bad", "fallback") == "fallback"


class TestSafeGetNested:
    def test_safe_get_nested_normal(self):
        obj = MagicMock()
        obj.a.b.c = "deep"
        assert safe_get_nested(obj, "a", "b", "c") == "deep"

    def test_safe_get_nested_missing(self):
        obj = MagicMock(spec=[])
        assert safe_get_nested(obj, "a", "b", default="nope") == "nope"

    def test_safe_get_nested_none_intermediate(self):
        """Returns default when intermediate attribute is None."""
        obj = MagicMock()
        obj.a = None
        assert safe_get_nested(obj, "a", "b", default="nope") == "nope"


class TestSetOptionalAttr:
    def test_set_optional_attr_none(self):
        span = MagicMock()
        set_optional_attr(span, "key", None)
        span.set_attribute.assert_not_called()

    def test_set_optional_attr_value(self):
        span = MagicMock()
        set_optional_attr(span, "key", "val")
        span.set_attribute.assert_called_once_with("key", "val")

    def test_set_optional_attr_truncates_long_string(self):
        span = MagicMock()
        long_val = "x" * (MAX_ATTR_LEN + 100)
        set_optional_attr(span, "key", long_val)
        call_args = span.set_attribute.call_args
        assert len(call_args[0][1]) == MAX_ATTR_LEN


class TestTruncateText:
    def test_truncate_none(self):
        assert truncate_text(None) is None

    def test_truncate_short(self):
        assert truncate_text("hello") == "hello"

    def test_truncate_long(self):
        long_val = "a" * 2000
        result = truncate_text(long_val)
        assert len(result) == MAX_ATTR_LEN


class TestJsonDumpsAttr:
    def test_json_dumps_dict(self):
        result = json_dumps_attr({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result


class TestGenaiMessages:
    def test_genai_messages_with_dicts(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = genai_messages(msgs)
        assert "user" in result
        assert "hello" in result

    def test_genai_messages_empty(self):
        result = genai_messages([])
        assert result == "[]"

    def test_genai_messages_none(self):
        result = genai_messages(None)
        assert result == "[]"


class TestInstrumentorMeta:
    def test_instrumentation_dependencies(self):
        from opentelemetry.instrumentation.slop_code import (
            SlopCodeInstrumentor,
        )

        instrumentor = SlopCodeInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert ("slop-code-bench >= 0.1",) == deps
