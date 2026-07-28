# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import (
    Any,
    Dict,
    Set,
    Union,
)

Primitive = Union[bool, str, int, float]
BoolList = list[bool]
StringList = list[str]
IntList = list[int]
FloatList = list[float]
HomogenousPrimitiveList = Union[BoolList, StringList, IntList, FloatList]
FlattenedValue = Union[Primitive, HomogenousPrimitiveList]
FlattenedDict = Dict[str, FlattenedValue]


_logger = logging.getLogger(__name__)


def _is_homogenous_primitive_list(v):
    if len(v) == 0:
        return True
    if not isinstance(v[0], (str, bool, int, float)):
        return False
    first_entry_value_type = type(v[0])
    for entry in v[1:]:
        if not isinstance(entry, first_entry_value_type):
            return False
    return True


def _flatten_compound_value_using_json(
    key: str,
    value: Any,
    exclude_keys: Set[str],
    _from_json=False,
) -> FlattenedDict:
    if _from_json:
        _logger.debug(
            "Cannot flatten value with key %s; value: %s", key, value
        )
        return {}
    try:
        json_string = json.dumps(value)
    except TypeError:
        _logger.debug(
            "Cannot flatten value with key %s; value: %s. Not JSON serializable.",
            key,
            value,
        )
        return {}
    return _flatten_value(
        key,
        json.loads(json_string),
        exclude_keys,
        # Ensure that we don't recurse indefinitely if "json.loads()" somehow returns
        # a complex, compound object that does not get handled by the "primitive", "list",
        # or "dict" cases. Prevents falling back on the JSON serialization fallback path.
        True,
    )


def _flatten_compound_value(
    key: str,
    value: Any,
    exclude_keys: Set[str],
    _from_json=False,
) -> FlattenedDict:
    if isinstance(value, dict):
        return flatten_dict(value, key, exclude_keys)
    if isinstance(value, list):
        if _is_homogenous_primitive_list(value):
            return {key: value}
        result = {f"{key}.length": len(value)}
        for idx, val in enumerate(value):
            result.update(_flatten_value(f"{key}[{idx}]", val, exclude_keys))
        return result
    if hasattr(value, "model_dump"):
        try:
            return flatten_dict(value.model_dump(), key, exclude_keys)
        except TypeError:
            return {key: str(value)}
    return _flatten_compound_value_using_json(
        key, value, exclude_keys, _from_json
    )


def _flatten_value(
    key: str,
    value: Any,
    exclude_keys: Set[str],
    _from_json=False,
) -> FlattenedDict:
    if value is None or key in exclude_keys:
        return {}
    if isinstance(value, (str, bool, int, float)):
        return {key: value}
    return _flatten_compound_value(key, value, exclude_keys, _from_json)


def flatten_dict(
    d: Dict[str, Any],
    key_prefix: str,
    exclude_keys: Set[str],
) -> FlattenedDict:
    result = {}
    for key, value in d.items():
        if key not in exclude_keys:
            result.update(
                _flatten_value(f"{key_prefix}.{key}", value, exclude_keys)
            )
    return result
