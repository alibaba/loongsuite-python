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

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai._suppression import suppress_http_instrumentation


def test_http_plugin_recognizes_suppression():
    httpx_instrumentation = pytest.importorskip(
        "opentelemetry.instrumentation.httpx"
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    plugin = httpx_instrumentation.HTTPXClientInstrumentor()
    plugin.instrument(tracer_provider=provider)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # pylint: disable=invalid-name
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with httpx.Client(trust_env=False) as client:
            with suppress_http_instrumentation():
                assert client.get(url + "/internal").status_code == 200
            assert len(exporter.get_finished_spans()) == 0
            client.get(url + "/business")
            assert len(exporter.get_finished_spans()) == 1

        async def run():
            async with httpx.AsyncClient(trust_env=False) as client:
                try:
                    with suppress_http_instrumentation():
                        await client.get(url + "/internal")
                        raise ValueError("failed processing")
                except ValueError:
                    pass
                assert len(exporter.get_finished_spans()) == 1
                await client.get(url + "/business")
                assert len(exporter.get_finished_spans()) == 2

        asyncio.run(run())
    finally:
        plugin.uninstrument()
        provider.shutdown()
        server.shutdown()
        server.server_close()
        thread.join()
