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

"""PyPI distribution list for LoongSuite releases (stdlib only).

Used by ``collect_loongsuite_changelog.py`` (release notes) and
``build_loongsuite_package.py`` (must stay in sync with ``build_pypi_packages``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Set

# FIXME: Omit from PyPI only; these still ship in the GitHub Release tar. Remove entries
# once each plugin is tested enough for standalone PyPI wheels.
PYPI_SKIP_INSTRUMENTATION_LOONGSUITE: frozenset[str] = frozenset(
    (
        "loongsuite-instrumentation-agno",
        "loongsuite-instrumentation-dify",
        "loongsuite-instrumentation-mcp",
    )
)

_DEFAULT_SKIP_CONFIG = (
    Path(__file__).resolve().parent / "loongsuite-build-config.json"
)


def load_skip_config(config_path: Path) -> Set[str]:
    """Load package directory names to skip from JSON config (``skip_packages``)."""
    if not config_path.exists():
        return set()
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return set(config.get("skip_packages", []))


def read_project_distribution_name(pyproject_path: Path) -> Optional[str]:
    """Read ``project.name`` from a minimal TOML scan (no third-party parser)."""
    text = pyproject_path.read_text(encoding="utf-8")
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if not in_project:
            continue
        m = re.match(
            r'^name\s*=\s*["\']([^"\']+)["\']',
            stripped,
        )
        if m:
            return m.group(1)
    return None


def list_pypi_distribution_names(
    base_dir: Path,
    skip_config_path: Optional[Path] = None,
) -> List[str]:
    """Names of distributions produced by ``build_pypi_packages`` for upload to PyPI."""
    base_dir = base_dir.resolve()
    cfg = skip_config_path or _DEFAULT_SKIP_CONFIG
    skip_packages = load_skip_config(cfg)
    names: List[str] = []

    util_genai_dir = base_dir / "util" / "opentelemetry-util-genai"
    if (
        util_genai_dir.is_dir()
        and (util_genai_dir / "pyproject.toml").is_file()
    ):
        names.append("loongsuite-otel-util-genai")

    distro_dir = base_dir / "loongsuite-distro"
    if distro_dir.is_dir() and (distro_dir / "pyproject.toml").is_file():
        names.append("loongsuite-distro")

    site_bootstrap_dir = base_dir / "loongsuite-site-bootstrap"
    if (
        site_bootstrap_dir.is_dir()
        and (site_bootstrap_dir / "pyproject.toml").is_file()
    ):
        names.append("loongsuite-site-bootstrap")

    inst_dir = base_dir / "instrumentation-loongsuite"
    if inst_dir.is_dir():
        for package_dir in sorted(inst_dir.iterdir()):
            if (
                not package_dir.is_dir()
                or not (package_dir / "pyproject.toml").is_file()
            ):
                continue
            pkg_dir_name = package_dir.name
            if pkg_dir_name in skip_packages:
                continue
            if pkg_dir_name in PYPI_SKIP_INSTRUMENTATION_LOONGSUITE:
                continue
            proj_name = read_project_distribution_name(
                package_dir / "pyproject.toml"
            )
            if proj_name:
                names.append(proj_name)

    return names
