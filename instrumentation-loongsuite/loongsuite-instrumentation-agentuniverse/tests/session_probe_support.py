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

"""State the session probe subprocess shares with its TelemetryManager config.

``TelemetryManager`` builds its span processor from an import path, so the
exporter has to live in a module that is *imported*. Defined in the probe script
itself, it would belong to ``__main__`` there while ``TelemetryManager``
imported a second copy, and the spans would be collected into a list no test can
see.
"""

from typing import Any

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

#: Every span the exporter was handed, oldest first.
SPANS: list = []


class RecordingExporter(SpanExporter):
    """Append every exported span to :data:`SPANS`."""

    def export(self, spans: Any) -> Any:
        SPANS.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None
