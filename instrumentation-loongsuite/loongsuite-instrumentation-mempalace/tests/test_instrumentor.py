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

"""
Smoke tests for MemPalace instrumentation.

MemPalace is not installed in the test environment, so we inject a stub
``mempalace`` package into ``sys.modules`` that mirrors the 9 anchor points.
The instrumentor should patch each anchor and emit the expected spans and
attributes per /apsara/semantic-conventions/arms_docs/trace/.
"""

import sys
import types
import unittest
from unittest.mock import patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


def _install_fake_mempalace():
    """Build a stub mempalace package with the 9 anchor points."""
    if "mempalace" in sys.modules:
        return sys.modules["mempalace"]

    pkg = types.ModuleType("mempalace")
    pkg.__path__ = []  # mark as package
    sys.modules["mempalace"] = pkg

    # mcp_server module
    mcp_server = types.ModuleType("mempalace.mcp_server")
    sys.modules["mempalace.mcp_server"] = mcp_server

    def handle_request(request):
        method = request.get("method", "")
        if method == "ping":
            return {"jsonrpc": "2.0", "id": request.get("id"), "result": {}}
        if method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            handler = TOOLS.get(name, {}).get("handler")
            if handler is None:
                return {"jsonrpc": "2.0", "id": request.get("id"), "error": {"code": -32601}}
            return {"jsonrpc": "2.0", "id": request.get("id"), "result": handler(**args)}
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": {}}

    def _resolve_kg_path():
        return "/tmp/fake_palace/kg.sqlite3"

    def _call_kg(op):
        # op is a lambda taking a kg dict
        return op({"triples": [{"s": "a", "p": "b", "o": "c"}]})

    def tool_search(query, limit=5, max_distance=1.5, wing=None, room=None):
        return {"drawers": [{"id": "d1"}, {"id": "d2"}], "count": 2}

    def tool_add_drawer(content, wing=None, added_by=None):
        return {"drawer_id": "dr-1", "count": 1}

    def tool_status():
        return {"total_drawers": 5}

    TOOLS = {
        "mempalace_search": {
            "description": "search drawers",
            "input_schema": {"type": "object"},
            "handler": tool_search,
        },
        "mempalace_add_drawer": {
            "description": "add a drawer",
            "input_schema": {"type": "object"},
            "handler": tool_add_drawer,
        },
        "mempalace_status": {
            "description": "palace status",
            "input_schema": {"type": "object"},
            "handler": tool_status,
        },
    }

    mcp_server.handle_request = handle_request
    mcp_server._resolve_kg_path = _resolve_kg_path
    mcp_server._call_kg = _call_kg
    mcp_server.TOOLS = TOOLS
    mcp_server.tool_search = tool_search
    mcp_server.tool_add_drawer = tool_add_drawer
    mcp_server.tool_status = tool_status
    pkg.mcp_server = mcp_server

    # searcher module
    searcher = types.ModuleType("mempalace.searcher")
    sys.modules["mempalace.searcher"] = searcher

    def search_memories(query, palace_path=None, n_results=5, max_distance=0.0, wing=None, room=None, collection_name=None):
        return {
            "ids": [["d1", "d2"]],
            "distances": [[0.1, 0.2]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{}, {}]],
        }

    def search(query, palace_path=None, n_results=5):
        return search_memories(query, palace_path=palace_path, n_results=n_results)

    searcher.search_memories = search_memories
    searcher.search = search
    pkg.searcher = searcher

    # backends.chroma module
    backends = types.ModuleType("mempalace.backends")
    backends.__path__ = []
    sys.modules["mempalace.backends"] = backends
    chroma = types.ModuleType("mempalace.backends.chroma")
    sys.modules["mempalace.backends.chroma"] = chroma

    class ChromaCollection:
        def __init__(self, collection=None, palace_path=None):
            self._collection = collection
            self._palace_path = palace_path
            self.name = "mempalace_drawers"

        def add(self, *, documents, ids, metadatas=None, embeddings=None):
            return None

        def upsert(self, *, documents, ids, metadatas=None, embeddings=None):
            return None

        def query(self, *, query_texts=None, query_embeddings=None, n_results=10, where=None):
            return {"ids": [["d1", "d2"]], "distances": [[0.1, 0.2]], "documents": [["a", "b"]]}

        def get(self, *, ids=None, where=None, limit=None):
            return {"ids": ["d1"], "documents": ["a"], "metadatas": [{}]}

        def delete(self, *, ids=None, where=None):
            return None

    chroma.ChromaCollection = ChromaCollection
    pkg.backends = backends
    backends.chroma = chroma

    # embedding module
    embedding = types.ModuleType("mempalace.embedding")
    sys.modules["mempalace.embedding"] = embedding

    def current_model_name(model=None):
        return "embeddinggemma-300m"

    def probe_dimension(device=None, model=None):
        return 768

    class EmbeddinggemmaONNX:
        def __call__(self, input):  # noqa: A002 - mirrors MemPalace EF protocol
            return [[0.1] * 768]

    embedding.current_model_name = current_model_name
    embedding.probe_dimension = probe_dimension
    embedding.EmbeddinggemmaONNX = EmbeddinggemmaONNX
    pkg.embedding = embedding

    # llm_client module
    llm_client = types.ModuleType("mempalace.llm_client")
    sys.modules["mempalace.llm_client"] = llm_client

    def _http_post_json(url, body, headers, timeout):
        return {
            "model": body.get("model", "test-model"),
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    llm_client._http_post_json = _http_post_json
    pkg.llm_client = llm_client

    # closet_llm module
    closet_llm = types.ModuleType("mempalace.closet_llm")
    sys.modules["mempalace.closet_llm"] = closet_llm

    class LLMConfig:
        def __init__(self, model="qwen3.6-plus", endpoint="https://dashscope.aliyuncs.com", key=""):
            self.model = model
            self.endpoint = endpoint
            self.key = key

    def _call_llm(cfg, source_file, wing, room, content):
        return {"fact": "extracted"}, {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}

    closet_llm.LLMConfig = LLMConfig
    closet_llm._call_llm = _call_llm
    pkg.closet_llm = closet_llm

    # service module
    service = types.ModuleType("mempalace.service")
    sys.modules["mempalace.service"] = service

    def execute_job(kind, payload):
        return {"success": True, "kind": kind}

    service.execute_job = execute_job
    pkg.service = service

    # miner module
    miner = types.ModuleType("mempalace.miner")
    sys.modules["mempalace.miner"] = miner

    def mine(project_dir, palace_path=None, wing_override=None, agent="mempalace", limit=0, dry_run=False):
        return {"output": "mined 5 files", "files": 5}

    miner.mine = mine
    pkg.miner = miner

    return pkg


class TestMemPalaceInstrumentor(unittest.TestCase):
    """Tests for the MemPalace instrumentor against a stub mempalace."""

    def setUp(self):
        _install_fake_mempalace()
        self.exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        trace.set_tracer_provider(self.tracer_provider)

        from opentelemetry.instrumentation.mempalace import (
            MemPalaceInstrumentor,
        )

        self.instrumentor = MemPalaceInstrumentor()

    def tearDown(self):
        try:
            self.instrumentor.uninstrument()
        except Exception:
            pass
        self.exporter.clear()

    def test_instrumentation_dependencies(self):
        deps = self.instrumentor.instrumentation_dependencies()
        self.assertIsInstance(deps, tuple)
        self.assertTrue(any("mempalace" in d for d in deps))

    def test_instrument_without_mempalace_is_noop(self):
        """If mempalace isn't importable, _instrument must not raise."""
        with patch.dict(sys.modules, {"mempalace": None}):
            # Force the inner `import mempalace` to fail by deleting the cached module
            saved = sys.modules.pop("mempalace", None)
            try:
                self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)  # should warn + return
            finally:
                if saved is not None:
                    sys.modules["mempalace"] = saved

    def test_mcp_server_span(self):
        import mempalace.mcp_server as ms

        self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
        result = ms.handle_request({"jsonrpc": "2.0", "id": "1", "method": "ping"})
        self.assertEqual(result["result"], {})

        spans = self.exporter.get_finished_spans()
        self.assertTrue(len(spans) >= 1)
        span = spans[-1]
        self.assertIn("ping", span.name)
        attrs = span.attributes
        self.assertEqual(attrs.get("gen_ai.span.kind"), "SERVER")
        self.assertEqual(attrs.get("gen_ai.operation.name"), "mcp.server")
        self.assertEqual(attrs.get("mcp.method.name"), "ping")
        self.assertEqual(attrs.get("network.protocol.version"), "2.0")

    def test_tool_memory_search_span(self):
        import mempalace.mcp_server as ms

        self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
        handler = ms.TOOLS["mempalace_search"]["handler"]
        result = handler(query="hello", limit=5)
        self.assertEqual(result["count"], 2)

        spans = self.exporter.get_finished_spans()
        names = [s.name for s in spans]
        self.assertTrue(any("mempalace_search" in n for n in names))
        span = next(s for s in spans if "mempalace_search" in s.name)
        attrs = span.attributes
        self.assertEqual(attrs.get("gen_ai.span.kind"), "TOOL")
        self.assertEqual(attrs.get("gen_ai.tool.name"), "mempalace_search")
        self.assertEqual(attrs.get("gen_ai.memory.operation"), "search")
        self.assertEqual(attrs.get("gen_ai.operation.name"), "memory_operation")
        self.assertEqual(attrs.get("gen_ai.memory.result_count"), 2)
        # capture-message-content default false → no tool.call.arguments
        self.assertNotIn("gen_ai.tool.call.arguments", attrs)

    def test_tool_memory_add_span(self):
        import mempalace.mcp_server as ms

        self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
        handler = ms.TOOLS["mempalace_add_drawer"]["handler"]
        result = handler(content="hello", added_by="agent-1")
        self.assertEqual(result["drawer_id"], "dr-1")

        spans = self.exporter.get_finished_spans()
        span = next(s for s in spans if "mempalace_add_drawer" in s.name)
        attrs = span.attributes
        self.assertEqual(attrs.get("gen_ai.memory.operation"), "add")
        self.assertEqual(attrs.get("gen_ai.memory.memory_type"), "procedural_memory")
        self.assertEqual(attrs.get("gen_ai.memory.agent_id"), "agent-1")
        # gen_ai.memory.id for add comes from result drawer_id (add has no
        # drawer_id input); execute.md requires it only for get/update/delete.
        self.assertEqual(attrs.get("gen_ai.memory.id"), "dr-1")

    def test_retriever_span(self):
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
        import mempalace.searcher as se

        se.search_memories("hello", palace_path="/tmp/palace", n_results=5)
        spans = self.exporter.get_finished_spans()
        span = next(s for s in spans if s.name.startswith("retrieval "))
        attrs = span.attributes
        self.assertEqual(attrs.get("gen_ai.span.kind"), "RETRIEVER")
        self.assertEqual(attrs.get("gen_ai.operation.name"), "retrieval")
        self.assertEqual(attrs.get("gen_ai.provider.name"), "chroma")
        self.assertEqual(attrs.get("gen_ai.request.model"), "embeddinggemma-300m")
        self.assertEqual(attrs.get("gen_ai.request.top_k"), 5.0)
        # capture off → no retrieval.query.text
        self.assertNotIn("gen_ai.retrieval.query.text", attrs)

    def test_llm_http_span(self):
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
        import mempalace.llm_client as lc

        lc._http_post_json(
            "https://dashscope.aliyuncs.com/v1/chat/completions",
            {"model": "qwen3.6-plus", "messages": [{"role": "user", "content": "hi"}]},
            {},
            30,
        )
        spans = self.exporter.get_finished_spans()
        span = next(s for s in spans if s.name.startswith("chat "))
        attrs = span.attributes
        self.assertEqual(attrs.get("gen_ai.span.kind"), "LLM")
        self.assertEqual(attrs.get("gen_ai.operation.name"), "chat")
        self.assertEqual(attrs.get("gen_ai.request.model"), "qwen3.6-plus")
        self.assertEqual(attrs.get("gen_ai.usage.input_tokens"), 5)
        self.assertEqual(attrs.get("gen_ai.usage.output_tokens"), 3)
        self.assertEqual(attrs.get("gen_ai.usage.total_tokens"), 8)
        self.assertEqual(attrs.get("gen_ai.response.finish_reasons"), ("stop",))

    def test_llm_closet_span(self):
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
        import mempalace.closet_llm as cl

        parsed, usage = cl._call_llm(cl.LLMConfig(), "src/x.py", "wing", "room", "content")
        spans = self.exporter.get_finished_spans()
        span = next(s for s in spans if s.name.startswith("chat "))
        attrs = span.attributes
        self.assertEqual(attrs.get("gen_ai.span.kind"), "LLM")
        self.assertEqual(attrs.get("gen_ai.request.model"), "qwen3.6-plus")
        self.assertEqual(attrs.get("gen_ai.usage.input_tokens"), 10)
        self.assertEqual(attrs.get("gen_ai.usage.total_tokens"), 17)

    def test_vector_subphase_default_off(self):
        """Vector sub-span is gated off by default."""
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
        import mempalace.backends.chroma as ch

        col = ch.ChromaCollection(collection=type("C", (), {"name": "mempalace_drawers", "metadata": {"hnsw:space": "cosine"}})(), palace_path="/tmp/palace")
        col.query(query_texts=["hi"], n_results=5)
        spans = self.exporter.get_finished_spans()
        self.assertEqual([s.name for s in spans], [])

    def test_vector_subphase_enabled(self):
        import os

        os.environ["OTEL_INSTRUMENTATION_MEMPALACE_INNER_ENABLED"] = "true"
        try:
            self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
            import mempalace.backends.chroma as ch

            col = ch.ChromaCollection(
                collection=type("C", (), {"name": "mempalace_drawers", "metadata": {"hnsw:space": "cosine"}})(),
                palace_path="/tmp/palace",
            )
            col.query(query_texts=["hi"], n_results=5, where={"wing": "x"})
            spans = self.exporter.get_finished_spans()
            span = next(s for s in spans if s.name == "chroma.query")
            attrs = span.attributes
            self.assertEqual(attrs.get("gen_ai.memory.inner_name"), "vector")
            self.assertEqual(attrs.get("gen_ai.memory.data_source.type"), "chroma")
            self.assertEqual(attrs.get("gen_ai.memory.vector.method"), "query")
            self.assertEqual(attrs.get("gen_ai.memory.vector.limit"), 5)
            self.assertEqual(attrs.get("gen_ai.memory.vector.metric_type"), "cosine")
            self.assertIn("wing", attrs.get("gen_ai.memory.vector.filter_keys", ()))
            self.assertEqual(attrs.get("gen_ai.memory.vector.result_count"), 2)
        finally:
            os.environ.pop("OTEL_INSTRUMENTATION_MEMPALACE_INNER_ENABLED", None)

    def test_graph_subphase_enabled(self):
        import os

        os.environ["OTEL_INSTRUMENTATION_MEMPALACE_INNER_ENABLED"] = "true"
        try:
            self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
            import mempalace.mcp_server as ms

            ms._call_kg(lambda kg: kg["triples"])
            spans = self.exporter.get_finished_spans()
            span = next(s for s in spans if s.name.startswith("sqlite."))
            attrs = span.attributes
            self.assertEqual(attrs.get("gen_ai.memory.inner_name"), "graph")
            self.assertEqual(attrs.get("gen_ai.memory.data_source.type"), "sqlite")
            self.assertEqual(attrs.get("gen_ai.memory.graph.method"), "call")
        finally:
            os.environ.pop("OTEL_INSTRUMENTATION_MEMPALACE_INNER_ENABLED", None)

    def test_task_span(self):
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
        import mempalace.service as svc

        svc.execute_job("mine", {"project_dir": "/tmp"})
        spans = self.exporter.get_finished_spans()
        span = next(s for s in spans if s.name == "run_task mine")
        attrs = span.attributes
        self.assertEqual(attrs.get("gen_ai.span.kind"), "TASK")
        self.assertEqual(attrs.get("gen_ai.operation.name"), "run_task")
        self.assertEqual(attrs.get("gen_ai.task.name"), "mine")

    def test_chain_span(self):
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
        import mempalace.miner as mn

        mn.mine("/tmp/project", palace_path="/tmp/palace", limit=10)
        spans = self.exporter.get_finished_spans()
        span = next(s for s in spans if s.name == "chain mine")
        attrs = span.attributes
        self.assertEqual(attrs.get("gen_ai.span.kind"), "CHAIN")
        self.assertEqual(attrs.get("gen_ai.operation.name"), "workflow")

    def test_uninstrument_restores_tools(self):
        import mempalace.mcp_server as ms

        original = ms.TOOLS["mempalace_search"]["handler"]
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
        wrapped = ms.TOOLS["mempalace_search"]["handler"]
        self.assertIsNot(original, wrapped)  # patched in place
        self.instrumentor.uninstrument()
        restored = ms.TOOLS["mempalace_search"]["handler"]
        # After uninstrument, calling should not produce spans
        restored(query="hello", limit=5)
        spans = self.exporter.get_finished_spans()
        self.assertEqual([s.name for s in spans], [])

    def test_capture_message_content(self):
        import os

        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
        try:
            self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
            import mempalace.mcp_server as ms

            handler = ms.TOOLS["mempalace_add_drawer"]["handler"]
            handler(content="secret content", added_by="agent-1")
            spans = self.exporter.get_finished_spans()
            span = next(s for s in spans if "mempalace_add_drawer" in s.name)
            attrs = span.attributes
            self.assertIn("gen_ai.tool.call.arguments", attrs)
            # WAL redact: content key value must be replaced
            args_text = attrs["gen_ai.tool.call.arguments"]
            self.assertIn("<redacted:len=", args_text)
            self.assertNotIn("secret content", args_text)
        finally:
            os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)

    def test_exception_records_error(self):
        import mempalace.mcp_server as ms

        def boom(**kwargs):
            raise RuntimeError("kaboom")

        ms.TOOLS["__test_boom__"] = {
            "description": "raises",
            "input_schema": {"type": "object"},
            "handler": boom,
        }
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, skip_dep_check=True)
        with self.assertRaises(RuntimeError):
            ms.TOOLS["__test_boom__"]["handler"]()

        spans = self.exporter.get_finished_spans()
        span = next(s for s in spans if "__test_boom__" in s.name)
        self.assertEqual(span.status.status_code.name, "ERROR")
        self.assertEqual(span.attributes.get("error.type"), "RuntimeError")
        del ms.TOOLS["__test_boom__"]


if __name__ == "__main__":
    unittest.main()
