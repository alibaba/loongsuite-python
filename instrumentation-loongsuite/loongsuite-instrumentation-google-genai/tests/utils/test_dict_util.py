# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from opentelemetry.instrumentation.google_genai import dict_util


class PydanticModel(BaseModel):
    """Used to verify handling of pydantic models in the flattener."""

    str_value: str = ""
    int_value: int = 0


class ModelDumpableNotPydantic:
    """Used to verify general handling of 'model_dump'."""

    def __init__(self, dump_output):
        self._dump_output = dump_output

    def model_dump(self):
        return self._dump_output


class NotJsonSerializable:
    def __init__(self):
        pass


def test_flatten_empty_dict():
    input_dict = {}
    output_dict = dict_util.flatten_dict(input_dict, "", set())
    assert output_dict is not None
    assert isinstance(output_dict, dict)
    assert not output_dict


def test_flatten_simple_dict():
    input_dict = {
        "int_key": 1,
        "string_key": "somevalue",
        "float_key": 3.14,
        "bool_key": True,
    }
    assert dict_util.flatten_dict(input_dict, "gcp", set()) == {
        "gcp.int_key": 1,
        "gcp.string_key": "somevalue",
        "gcp.float_key": 3.14,
        "gcp.bool_key": True,
    }


def test_flatten_nested_dict():
    input_dict = {
        "int_key": 1,
        "string_key": "somevalue",
        "float_key": 3.14,
        "bool_key": True,
        "object_key": {
            "nested": {
                "foo": 1,
                "bar": "baz",
            },
            "qux": 54321,
        },
    }
    assert dict_util.flatten_dict(input_dict, "gcp", set()) == {
        "gcp.int_key": 1,
        "gcp.string_key": "somevalue",
        "gcp.float_key": 3.14,
        "gcp.bool_key": True,
        "gcp.object_key.nested.foo": 1,
        "gcp.object_key.nested.bar": "baz",
        "gcp.object_key.qux": 54321,
    }


def test_flatten_with_key_exclusion():
    input_dict = {
        "int_key": 1,
        "string_key": "somevalue",
        "float_key": 3.14,
        "bool_key": True,
    }
    output = dict_util.flatten_dict(input_dict, "gcp", {"gcp.int_key"})
    assert "gcp.int_key" not in output
    assert output == {
        "gcp.string_key": "somevalue",
        "gcp.float_key": 3.14,
        "gcp.bool_key": True,
    }


def test_flatten_with_prefixing():
    input_dict = {
        "int_key": 1,
        "string_key": "somevalue",
        "float_key": 3.14,
        "bool_key": True,
    }
    output = dict_util.flatten_dict(input_dict, "gcp.someprefix", set())
    assert output == {
        "gcp.someprefix.int_key": 1,
        "gcp.someprefix.string_key": "somevalue",
        "gcp.someprefix.float_key": 3.14,
        "gcp.someprefix.bool_key": True,
    }


def test_flatten_with_pydantic_model_value():
    input_dict = {
        "foo": PydanticModel(str_value="bar", int_value=123),
    }

    output = dict_util.flatten_dict(input_dict, "gcp", set())
    assert output == {
        "gcp.foo.str_value": "bar",
        "gcp.foo.int_value": 123,
    }
    assert dict_util.flatten_dict({"foo": PydanticModel}, "gcp", set()) == {
        "gcp.foo": "<class 'tests.utils.test_dict_util.PydanticModel'>"
    }


def test_flatten_with_model_dumpable_value():
    input_dict = {
        "foo": ModelDumpableNotPydantic(
            {
                "str_value": "bar",
                "int_value": 123,
            }
        ),
    }

    output = dict_util.flatten_dict(input_dict, "gcp", set())
    assert output == {
        "gcp.foo.str_value": "bar",
        "gcp.foo.int_value": 123,
    }


def test_flatten_with_mixed_structures():
    input_dict = {
        "foo": ModelDumpableNotPydantic(
            {
                "pydantic": PydanticModel(str_value="bar", int_value=123),
            }
        ),
    }

    output = dict_util.flatten_dict(input_dict, "gcp", set())
    assert output == {
        "gcp.foo.pydantic.str_value": "bar",
        "gcp.foo.pydantic.int_value": 123,
    }


def test_converts_tuple_with_json_fallback():
    input_dict = {
        "foo": ("abc", 123),
    }
    output = dict_util.flatten_dict(input_dict, "gcp", set())
    assert output == {
        "gcp.foo.length": 2,
        "gcp.foo[0]": "abc",
        "gcp.foo[1]": 123,
    }


def test_json_conversion_handles_unicode():
    input_dict = {
        "foo": ("❤️", 123),
    }
    output = dict_util.flatten_dict(input_dict, "gcp", set())
    assert output == {
        "gcp.foo.length": 2,
        "gcp.foo[0]": "❤️",
        "gcp.foo[1]": 123,
    }


def test_flatten_with_complex_object_not_json_serializable():
    result = dict_util.flatten_dict(
        {
            "cannot_serialize_directly": NotJsonSerializable(),
        },
        "",
        set(),
    )
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) == 0


def test_flatten_good_with_non_serializable_complex_object():
    result = dict_util.flatten_dict(
        {
            "foo": {
                "bar": "blah",
                "baz": 5,
            },
            "cannot_serialize_directly": NotJsonSerializable(),
        },
        "gcp",
        set(),
    )
    assert result == {
        "gcp.foo.bar": "blah",
        "gcp.foo.baz": 5,
    }


def test_flatten_simple_homogenous_primitive_string_list():
    input_dict = {"list_value": ["abc", "def"]}
    assert dict_util.flatten_dict(input_dict, "gcp", set()) == {
        "gcp.list_value": ["abc", "def"],
    }


def test_flatten_simple_homogenous_primitive_int_list():
    input_dict = {"list_value": [123, 456]}
    assert dict_util.flatten_dict(input_dict, "gcp", set()) == {
        "gcp.list_value": [123, 456],
    }


def test_flatten_simple_homogenous_primitive_bool_list():
    input_dict = {"list_value": [True, False]}
    assert dict_util.flatten_dict(input_dict, "gcp", set()) == {
        "gcp.list_value": [True, False],
    }


def test_flatten_simple_heterogenous_primitive_list():
    input_dict = {"list_value": ["abc", 123]}
    assert dict_util.flatten_dict(input_dict, "gcp", set()) == {
        "gcp.list_value.length": 2,
        "gcp.list_value[0]": "abc",
        "gcp.list_value[1]": 123,
    }


def test_flatten_list_of_compound_types():
    input_dict = {
        "list_value": [
            {"a": 1, "b": 2},
            {"x": 100, "y": 123, "z": 321},
            "blah",
            [
                "abc",
                123,
            ],
        ]
    }
    assert dict_util.flatten_dict(input_dict, "gcp", set()) == {
        "gcp.list_value.length": 4,
        "gcp.list_value[0].a": 1,
        "gcp.list_value[0].b": 2,
        "gcp.list_value[1].x": 100,
        "gcp.list_value[1].y": 123,
        "gcp.list_value[1].z": 321,
        "gcp.list_value[2]": "blah",
        "gcp.list_value[3].length": 2,
        "gcp.list_value[3][0]": "abc",
        "gcp.list_value[3][1]": 123,
    }
