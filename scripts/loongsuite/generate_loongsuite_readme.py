#!/usr/bin/env python3
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

import logging
import os
from pathlib import Path

import tomli

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loongsuite_readme_generator")

_prefix = "loongsuite-instrumentation-"

header = """
| Instrumentation | Supported Packages | Metrics support | Semconv status |
| --------------- | ------------------ | --------------- | -------------- |"""


def main(base_instrumentation_path):
    table = [header]
    for instrumentation in sorted(os.listdir(base_instrumentation_path)):
        instrumentation_path = os.path.join(
            base_instrumentation_path, instrumentation
        )
        if not os.path.isdir(
            instrumentation_path
        ) or not instrumentation.startswith(_prefix):
            continue

        pyproject_toml = Path(instrumentation_path) / "pyproject.toml"
        if not pyproject_toml.exists():
            continue

        try:
            with open(pyproject_toml, "rb") as f:
                pyproject = tomli.load(f)

            project = pyproject.get("project", {})
            readme_config = (
                pyproject.get("tool", {})
                .get("loongsuite", {})
                .get("readme", {})
            )
            optional_deps = project.get("optional-dependencies", {})
            instruments = optional_deps.get("instruments", [])
            instruments_any = optional_deps.get("instruments-any", [])

            # Extract package name from instrumentation directory name
            # e.g., "loongsuite-instrumentation-agentscope" -> "agentscope"
            name = instrumentation.replace(_prefix, "")

            instruments_display = readme_config.get("supported-packages", [])
            if instruments_display:
                instruments_all = tuple(instruments_display)
            elif not instruments and not instruments_any:
                instruments_all = (name,)
            else:
                instruments_all = tuple(instruments + instruments_any)

            # Try to get metrics support and semconv status from pyproject.toml
            # These might not be present, so use defaults
            supports_metrics = project.get("supports_metrics", False)
            semconv_status = project.get("semconv_status", "development")

            metric_column = "Yes" if supports_metrics else "No"

            supported_packages = "; ".join(instruments_all)
            table.append(
                f"| [{instrumentation}](./{instrumentation}) | {supported_packages} | {metric_column} | {semconv_status}"
            )
        except Exception as e:
            logger.warning(f"Failed to process {instrumentation}: {e}")
            continue

    readme_path = os.path.join(base_instrumentation_path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(table))
    logger.info(f"Generated {readme_path}")


if __name__ == "__main__":
    root_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    instrumentation_path = os.path.join(
        root_path, "instrumentation-loongsuite"
    )
    if os.path.exists(instrumentation_path):
        main(instrumentation_path)
    else:
        logger.warning(
            f"Instrumentation path does not exist: {instrumentation_path}"
        )
