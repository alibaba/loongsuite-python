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

"""Environment carrier for OpenTelemetry subprocess context propagation."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Optional

from opentelemetry.propagators.textmap import Getter, Setter


class EnvironmentGetter(Getter[Mapping[str, str]]):
    def __init__(self) -> None:
        self.carrier = {
            key.lower(): value for key, value in os.environ.items()
        }

    def get(self, carrier: Mapping[str, str], key: str) -> Optional[list[str]]:
        del carrier
        value = self.carrier.get(key.lower())
        return None if value is None else [value]

    def keys(self, carrier: Mapping[str, str]) -> list[str]:
        del carrier
        return list(self.carrier)


class EnvironmentSetter(Setter[MutableMapping[str, str]]):
    def set(
        self, carrier: MutableMapping[str, str], key: str, value: str
    ) -> None:
        carrier[key.upper()] = value
