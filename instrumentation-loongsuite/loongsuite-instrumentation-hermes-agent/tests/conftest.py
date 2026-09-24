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

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

HERMES_ROOT = Path(__file__).resolve().parents[4]
HERMES_AGENT_ROOT = HERMES_ROOT / "hermes-agent"
PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
HERMES_VENV_LIB = HERMES_AGENT_ROOT / ".venv" / "lib"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
LOCAL_MCP_SERVER = EXAMPLES_DIR / "local_demo_mcp_server.py"
DEMO_KB = EXAMPLES_DIR / "demo_knowledge_base.json"
PUBLIC_FIXTURE_ROOT = Path("/tmp/loongsuite-hermes-agent-tests")
PUBLIC_READ_FILE_PATH = PUBLIC_FIXTURE_ROOT / "read_file_input.txt"
PUBLIC_MULTI_AGENT_PATH = PUBLIC_FIXTURE_ROOT / "multi_agent_input.txt"
PUBLIC_WORKSPACE_PATH = PUBLIC_FIXTURE_ROOT / "workspace"

if "DASHSCOPE_API_KEY" not in os.environ:
    os.environ["DASHSCOPE_API_KEY"] = "test_dashscope_api_key"

if HERMES_VENV_LIB.exists():
    current_py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for candidate in HERMES_VENV_LIB.glob(f"{current_py_tag}/site-packages"):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

for path in (str(HERMES_AGENT_ROOT), str(PACKAGE_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

MeterProvider = importlib.import_module(
    "opentelemetry.sdk.metrics"
).MeterProvider
InMemoryMetricReader = importlib.import_module(
    "opentelemetry.sdk.metrics.export"
).InMemoryMetricReader
TracerProvider = importlib.import_module(
    "opentelemetry.sdk.trace"
).TracerProvider
SimpleSpanProcessor = importlib.import_module(
    "opentelemetry.sdk.trace.export"
).SimpleSpanProcessor
InMemorySpanExporter = importlib.import_module(
    "opentelemetry.sdk.trace.export.in_memory_span_exporter"
).InMemorySpanExporter

_MODULE = importlib.import_module("opentelemetry.instrumentation.hermes_agent")
HermesAgentInstrumentor = _MODULE.HermesAgentInstrumentor


@pytest.fixture(scope="function")
def instrumentation_module():
    return _MODULE


def pytest_configure(config: pytest.Config):
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"
    os.environ.setdefault(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
        "SPAN_ONLY",
    )
    config.option.api_key = os.getenv(
        "DASHSCOPE_API_KEY", "test_dashscope_api_key"
    )


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="metric_reader")
def fixture_metric_reader():
    reader = InMemoryMetricReader()
    yield reader


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(metric_reader):
    return MeterProvider(metric_readers=[metric_reader])


@pytest.fixture(scope="function")
def instrument(tracer_provider, meter_provider, span_exporter):
    instrumentor = HermesAgentInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
    yield instrumentor
    instrumentor.uninstrument()
    span_exporter.clear()


@pytest.fixture(scope="function")
def build_agent():
    def _build_agent(
        *,
        enabled_toolsets=None,
        max_iterations=4,
        skip_memory=True,
        session_db=None,
        reload_mcp=False,
    ):
        if reload_mcp:
            try:
                mcp_tool_module = importlib.import_module("tools.mcp_tool")
            except ImportError:
                pass
            else:
                mcp_tool_module.shutdown_mcp_servers()
                mcp_tool_module.discover_mcp_tools()

        ai_agent_class = importlib.import_module("run_agent").AIAgent

        agent = ai_agent_class(
            model="qwen-turbo",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            provider="dashscope",
            quiet_mode=True,
            skip_memory=skip_memory,
            skip_context_files=True,
            enabled_toolsets=enabled_toolsets
            if enabled_toolsets is not None
            else [],
            max_iterations=max_iterations,
            session_db=session_db,
        )
        return agent

    return _build_agent


@pytest.fixture(scope="function")
def require_live_hermes_env(monkeypatch):
    try:
        importlib.import_module("jiter.jiter")
    except ModuleNotFoundError:
        pytest.skip(
            "Hermes integration tests require optional runtime dependency "
            "'jiter.jiter'."
        )

    run_agent_module = importlib.import_module("run_agent")
    model_metadata_module = importlib.import_module("agent.model_metadata")

    monkeypatch.setattr(
        run_agent_module,
        "fetch_model_metadata",
        lambda: {},
        raising=False,
    )
    monkeypatch.setattr(
        model_metadata_module,
        "fetch_model_metadata",
        lambda: {},
        raising=False,
    )


@pytest.fixture(scope="function")
def fixture_path():
    return FIXTURES_DIR


def _publish_fixture(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return str(dest)


@pytest.fixture(scope="function")
def public_fixture_paths(fixture_path):
    PUBLIC_WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)
    return {
        "read_file": _publish_fixture(
            fixture_path / "read_file_input.txt",
            PUBLIC_READ_FILE_PATH,
        ),
        "multi_agent": _publish_fixture(
            fixture_path / "multi_agent_input.txt",
            PUBLIC_MULTI_AGENT_PATH,
        ),
        "workspace": str(PUBLIC_WORKSPACE_PATH),
    }


def _write_hermes_config(home: Path, *, enable_mcp: bool) -> None:
    home.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {
        "memory": {
            "memory_enabled": False,
            "user_profile_enabled": False,
        }
    }
    if enable_mcp:
        config["mcp_servers"] = {
            "demo": {
                "command": sys.executable,
                "args": [str(LOCAL_MCP_SERVER)],
                "env": {
                    "HERMES_DEMO_KB_PATH": str(DEMO_KB),
                    "LOONGSUITE_PYTHON_SITE_BOOTSTRAP": "False",
                },
            }
        }
    (home / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


@pytest.fixture(scope="function")
def isolated_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    _write_hermes_config(home, enable_mcp=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture(scope="function")
def local_demo_mcp_home(tmp_path, monkeypatch):
    pytest.importorskip("mcp", reason="MCP SDK not installed")
    home = tmp_path / ".hermes"
    _write_hermes_config(home, enable_mcp=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    mcp_tool_module = importlib.import_module("tools.mcp_tool")

    mcp_tool_module.shutdown_mcp_servers()
    mcp_tool_module.discover_mcp_tools()
    yield home
    mcp_tool_module.shutdown_mcp_servers()


def extract_metric_points(metric_reader, metric_name: str):
    metrics_data = metric_reader.get_metrics_data()
    if not metrics_data:
        return []

    points = []
    for resource_metrics in metrics_data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != metric_name:
                    continue
                points.extend(getattr(metric.data, "data_points", []))
    return points


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            ("authorization", "Bearer test_dashscope_api_key"),
            ("x-dashscope-api-key", "test_dashscope_api_key"),
            ("api-key", "test_dashscope_api_key"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers,
    }


class LiteralBlockScalar(str):
    """Formats strings as YAML literal blocks."""


def literal_block_scalar_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralBlockScalar, literal_block_scalar_presenter)


def process_string_value(string_value):
    try:
        json_data = json.loads(string_value)
        return LiteralBlockScalar(
            json.dumps(json_data, indent=2, ensure_ascii=False)
        )
    except (ValueError, TypeError):
        if isinstance(string_value, str) and len(string_value) > 80:
            return LiteralBlockScalar(string_value)
    return string_value


def convert_body_to_literal(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "body" and isinstance(value, dict) and "string" in value:
                value["string"] = process_string_value(value["string"])
            elif key == "body" and isinstance(value, str):
                data[key] = process_string_value(value)
            else:
                convert_body_to_literal(value)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            data[idx] = convert_body_to_literal(item)
    return data


class PrettyPrintJSONBody:
    @staticmethod
    def serialize(cassette_dict):
        cassette_dict = convert_body_to_literal(cassette_dict)
        return yaml.dump(
            cassette_dict, default_flow_style=False, allow_unicode=True
        )

    @staticmethod
    def deserialize(cassette_string):
        return yaml.safe_load(cassette_string)


@pytest.fixture(scope="module", autouse=True)
def fixture_vcr(request):
    try:
        vcr = request.getfixturevalue("vcr")
    except pytest.FixtureLookupError:
        return None

    vcr.register_serializer("yaml", PrettyPrintJSONBody)
    return vcr


def scrub_response_headers(response):
    headers = response.get("headers", {})
    for header in ("x-dashscope-request-id", "x-request-id"):
        if header in headers:
            headers[header] = "test_request_id"
    return response
