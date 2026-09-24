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

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "generate_loongsuite_bootstrap.py"
SPEC = importlib.util.spec_from_file_location(
    "generate_loongsuite_bootstrap",
    SCRIPT_PATH,
)
generate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate)


def test_registry_module_name_normalizes_package_name():
    assert (
        generate._registry_module_name("loongsuite-instrumentation-crewai")
        == "loongsuite_instrumentation_crewai"
    )
    assert (
        generate._registry_module_name(
            "opentelemetry-instrumentation-aiohttp-client"
        )
        == "opentelemetry_instrumentation_aiohttp_client"
    )


def test_registry_entry_combines_instruments_and_marks_default():
    entry = generate._registry_entry(
        {
            "name": "loongsuite-instrumentation-crewai",
            "bootstrap-source": "loongsuite",
            "requirement": "loongsuite-instrumentation-crewai==0.1.0",
            "instruments": ["crewai >= 0.100.0"],
            "instruments-any": ["crewai-tools >= 0.40.0"],
        }
    )

    assert entry == {
        "source": "loongsuite",
        "package": "loongsuite-instrumentation-crewai",
        "instrumentation": "loongsuite-instrumentation-crewai==0.1.0",
        "libraries": ["crewai >= 0.100.0", "crewai-tools >= 0.40.0"],
        "default": False,
    }

    default_entry = generate._registry_entry(
        {
            "name": "loongsuite-instrumentation-dashscope",
            "bootstrap-source": "loongsuite",
            "requirement": "loongsuite-instrumentation-dashscope==0.1.0",
        }
    )

    assert default_entry == {
        "source": "loongsuite",
        "package": "loongsuite-instrumentation-dashscope",
        "instrumentation": "loongsuite-instrumentation-dashscope==0.1.0",
        "libraries": [],
        "default": True,
    }


def test_render_registry_module_contains_local_metadata():
    source = generate._render_registry_module(
        {
            "source": "loongsuite",
            "package": "loongsuite-instrumentation-crewai",
            "instrumentation": "loongsuite-instrumentation-crewai==0.1.0",
            "libraries": ["crewai >= 0.100.0"],
            "default": False,
        },
        "# license",
        upstream_version="0.62b0",
        loongsuite_version="0.1.0",
    )

    assert "REGISTRY = {" in source
    assert '"source": "loongsuite"' in source
    assert '"package": "loongsuite-instrumentation-crewai"' in source
    assert "--upstream-version: 0.62b0" in source
    assert "--loongsuite-version: 0.1.0" in source


def test_registry_loader_builds_bootstrap_gen_compatible_lists(tmp_path):
    package_name = "test_bootstrap_registry"
    package_path = tmp_path / package_name
    package_path.mkdir()
    (package_path / "__init__.py").write_text(
        generate._registry_init_template.replace("{header}", "# license"),
        encoding="utf-8",
    )
    (package_path / "loongsuite_instrumentation_crewai.py").write_text(
        "REGISTRY = {\n"
        "    'source': 'loongsuite',\n"
        "    'package': 'loongsuite-instrumentation-crewai',\n"
        "    'instrumentation': 'loongsuite-instrumentation-crewai==0.1.0',\n"
        "    'libraries': ['crewai >= 0.100.0'],\n"
        "    'default': False,\n"
        "}\n",
        encoding="utf-8",
    )
    (package_path / "loongsuite_instrumentation_dashscope.py").write_text(
        "REGISTRY = {\n"
        "    'source': 'loongsuite',\n"
        "    'package': 'loongsuite-instrumentation-dashscope',\n"
        "    'instrumentation': 'loongsuite-instrumentation-dashscope==0.1.0',\n"
        "    'libraries': [],\n"
        "    'default': True,\n"
        "}\n",
        encoding="utf-8",
    )
    (
        package_path / "opentelemetry_instrumentation_aiohttp_client.py"
    ).write_text(
        "REGISTRY = {\n"
        "    'source': 'upstream',\n"
        "    'package': 'opentelemetry-instrumentation-aiohttp-client',\n"
        "    'instrumentation': "
        "'opentelemetry-instrumentation-aiohttp-client==0.62b0.dev',\n"
        "    'libraries': ['aiohttp ~= 3.0'],\n"
        "    'default': False,\n"
        "}\n",
        encoding="utf-8",
    )

    spec = importlib.util.spec_from_file_location(
        package_name,
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path)],
    )
    registry = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = registry
    try:
        spec.loader.exec_module(registry)
    finally:
        for module_name in list(sys.modules):
            if module_name == package_name or module_name.startswith(
                f"{package_name}."
            ):
                del sys.modules[module_name]

    assert registry.libraries == [
        {
            "library": "aiohttp ~= 3.0",
            "instrumentation": (
                "opentelemetry-instrumentation-aiohttp-client==0.62b0.dev"
            ),
        },
        {
            "library": "crewai >= 0.100.0",
            "instrumentation": "loongsuite-instrumentation-crewai==0.1.0",
        },
    ]
    assert registry.default_instrumentations == [
        "loongsuite-instrumentation-dashscope==0.1.0"
    ]
