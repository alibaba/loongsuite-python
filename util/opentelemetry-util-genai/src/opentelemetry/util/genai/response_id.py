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

"""Helpers for resolving provider and framework response identifiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _normalize_response_id(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    normalized = str(value).strip()
    return normalized or None


def extract_response_id(
    response: Any,
    *,
    fields: Sequence[str] = ("id",),
) -> str | None:
    """Extract the first non-empty identifier from ``response``.

    ``response`` may be an identifier itself, a mapping, or an SDK response
    object. Field order is deliberately caller-controlled because providers
    use different names for the same operation identifier. Transport-only
    fields such as OpenAI's ``_request_id`` are not considered implicitly.
    """

    direct_identifier = _normalize_response_id(response)
    if direct_identifier is not None:
        return direct_identifier

    for field in fields:
        try:
            candidate = (
                response.get(field)
                if isinstance(response, Mapping)
                else getattr(response, field, None)
            )
        except Exception:  # pylint: disable=broad-exception-caught
            # Provider SDK properties may raise arbitrary lazy-load errors;
            # response-ID telemetry must never break the model call.
            continue
        identifier = _normalize_response_id(candidate)
        if identifier is not None:
            return identifier
    return None


def resolve_response_id(
    provider_response: Any = None,
    framework_response: Any = None,
    *,
    provider_fields: Sequence[str] = ("id",),
    framework_fields: Sequence[str] = ("id",),
) -> str | None:
    """Prefer a provider identifier and fall back to the framework response."""

    return extract_response_id(
        provider_response,
        fields=provider_fields,
    ) or extract_response_id(
        framework_response,
        fields=framework_fields,
    )
