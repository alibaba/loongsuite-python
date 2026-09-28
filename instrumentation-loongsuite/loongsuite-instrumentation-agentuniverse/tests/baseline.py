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

"""Helpers that read telemetry and compare it with the framework's own.

The tests run the same fixture twice -- once under agentUniverse's own Agent,
LLM and Tool instrumentors and once under this package -- and compare the two
runs. Values that cannot be equal across processes (durations, generated ids)
are compared by type and sign instead of by value; everything else is compared
literally.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

AU_PREFIX = "au."
GEN_AI_PREFIX = "gen_ai."

#: Attributes whose value differs between two runs of the same call.
DYNAMIC_SUFFIXES = (
    "duration",
    "pair_id",
    "detail_tokens",
    "time_to_first_token",
)
DYNAMIC_EXACT = (
    "au.agent.first_token.duration",
    "au.llm.first_token.duration",
)

#: ``au.*.error.message`` embeds a traceback, whose line numbers are stable for
#: a given call but whose absolute paths are not interesting; compare the tail.
_LAST_LINE = re.compile(r"\n\s*")


def is_dynamic_attribute(key: str) -> bool:
    if key in DYNAMIC_EXACT:
        return True
    return any(key.endswith(f".{suffix}") for suffix in DYNAMIC_SUFFIXES)


def normalise_attribute_value(key: str, value: Any) -> Any:
    """Reduce one attribute to the part two runs can be expected to share."""
    if not is_dynamic_attribute(key):
        if (
            isinstance(value, str)
            and "Traceback (most recent call last)" in value
        ):
            return _LAST_LINE.split(value.strip())[-1]
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return f"number>={0}" if value >= 0 else "number<0"
    if isinstance(value, str):
        return "non-empty-string" if value else "empty-string"
    return f"type:{type(value).__name__}"


@dataclass(frozen=True)
class DataPoint:
    """One metric data point, flattened for assertions."""

    value: Any
    count: int
    labels: Dict[str, Any]


def _point_value(point: Any) -> Any:
    if hasattr(point, "value") and not hasattr(point, "sum"):
        return point.value
    if hasattr(point, "sum"):
        return point.sum
    return getattr(point, "value", None)


def collect_metrics(reader: Any) -> Dict[str, List[DataPoint]]:
    """Every data point the reader holds, grouped by metric name."""
    metrics: Dict[str, List[DataPoint]] = {}
    data = reader.get_metrics_data()
    if data is None:
        return metrics
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                points = metrics.setdefault(metric.name, [])
                for point in metric.data.data_points:
                    points.append(
                        DataPoint(
                            value=_point_value(point),
                            count=getattr(point, "count", 1),
                            labels=dict(point.attributes),
                        )
                    )
    return metrics


def points_for(
    metrics: Dict[str, List[DataPoint]], name: str, **labels: Any
) -> List[DataPoint]:
    """The points of one metric whose labels contain ``labels``."""
    return [
        point
        for point in metrics.get(name, [])
        if all(point.labels.get(key) == value for key, value in labels.items())
    ]


def counter_value(
    metrics: Dict[str, List[DataPoint]], name: str, **labels: Any
) -> float:
    """The value of a counter, summing the points that match ``labels``."""
    return sum(point.value for point in points_for(metrics, name, **labels))


def histogram_values(
    metrics: Dict[str, List[DataPoint]], name: str, **labels: Any
) -> List[Any]:
    return [point.value for point in points_for(metrics, name, **labels)]


def metric_names(metrics: Dict[str, List[DataPoint]]) -> set:
    return set(metrics)


@dataclass(frozen=True)
class SpanRecord:
    """The parts of a finished span the matrix compares."""

    name: str
    kind: str
    status: str
    description: str
    parent: Optional[str]
    attributes: Dict[str, Any]

    def prefix(self, prefix: str) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.attributes.items()
            if key.startswith(prefix)
        }

    @property
    def au(self) -> Dict[str, Any]:
        return self.prefix(AU_PREFIX)

    @property
    def gen_ai(self) -> Dict[str, Any]:
        return self.prefix(GEN_AI_PREFIX)


def snapshot(spans: Iterable[Any]) -> List[SpanRecord]:
    """Turn finished spans into comparable records, children after parents."""
    spans = list(spans)
    names = {span.context.span_id: span.name for span in spans}
    return [
        SpanRecord(
            name=span.name,
            kind=span.kind.name,
            status=span.status.status_code.name,
            description=span.status.description or "",
            parent=names.get(span.parent.span_id) if span.parent else None,
            attributes=dict(span.attributes or {}),
        )
        for span in spans
    ]


def find(records: Sequence[SpanRecord], name: str) -> SpanRecord:
    matches = [record for record in records if record.name == name]
    assert len(matches) == 1, f"expected one {name!r} span, got {len(matches)}"
    return matches[0]


def names(records: Sequence[SpanRecord]) -> List[str]:
    return [record.name for record in records]


def assert_same_au_contract(
    native: SpanRecord, loongsuite: SpanRecord, ignore: Sequence[str] = ()
) -> None:
    """The ``au.*`` contract of both implementations must line up.

    The attribute key set has to match exactly, and every value that cannot
    differ between two identical runs has to match too. Only the keys listed in
    ``ignore`` are allowed to differ in value.
    """
    native_au = native.au
    loongsuite_au = loongsuite.au
    assert set(native_au) == set(loongsuite_au), (
        f"{native.name}: au.* key sets differ; "
        f"only native: {sorted(set(native_au) - set(loongsuite_au))}, "
        f"only LoongSuite: {sorted(set(loongsuite_au) - set(native_au))}"
    )
    for key in sorted(native_au):
        expected = normalise_attribute_value(key, native_au[key])
        actual = normalise_attribute_value(key, loongsuite_au[key])
        if key in ignore:
            assert actual is not None or expected is None
            continue
        assert expected == actual, (
            f"{native.name}: {key} differs ({native_au[key]!r} vs {loongsuite_au[key]!r})"
        )


def assert_same_span_shape(
    native: SpanRecord, loongsuite: SpanRecord, ignore_au: Sequence[str] = ()
) -> None:
    assert native.kind == loongsuite.kind, native.name
    assert native.status == loongsuite.status, native.name
    assert native.parent == loongsuite.parent, (
        f"{native.name}: parent differs ({native.parent!r} vs {loongsuite.parent!r})"
    )
    assert_same_au_contract(native, loongsuite, ignore=ignore_au)
