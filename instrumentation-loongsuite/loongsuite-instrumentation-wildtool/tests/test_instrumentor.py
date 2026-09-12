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

"""Tests for WildToolInstrumentor lifecycle."""

from opentelemetry.instrumentation.wildtool import WildToolInstrumentor


class TestWildToolInstrumentor:
    def test_instrument_and_uninstrument(self, tracer_provider):
        instrumentor = WildToolInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True,
        )
        assert instrumentor._handler is not None
        instrumentor.uninstrument()
        assert instrumentor._handler is None

    def test_instrumentation_dependencies(self):
        instrumentor = WildToolInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert ("openai >= 1.0.0",) == deps
