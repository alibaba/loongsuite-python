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

"""Keep only wheels whose PyPI projects do not exist yet."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from loongsuite_pypi_manifest import list_pypi_distribution_names


def normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def distribution_name_from_wheel(wheel_path: Path) -> str:
    return wheel_path.name.split("-", 1)[0].replace("_", "-")


def project_exists_on_pypi(distribution_name: str, *, timeout: float) -> bool:
    normalized = normalize_distribution_name(distribution_name)
    url = f"https://pypi.org/pypi/{urllib.parse.quote(normalized)}/json"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "loongsuite-python-agent-release/1.0"},
    )
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return 200 <= response.status < 300
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                return False
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt == 3:
                raise RuntimeError(
                    f"PyPI project lookup failed for {distribution_name} "
                    f"at {url}: HTTP {exc.code}"
                ) from exc
        except OSError as exc:
            if attempt == 3:
                raise RuntimeError(
                    f"PyPI project lookup failed for {distribution_name} "
                    f"at {url}: {exc}"
                ) from exc

        time.sleep(attempt)


def write_github_outputs(output_path: Path, values: dict[str, str]) -> None:
    with output_path.open("a", encoding="utf-8") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def write_step_summary(
    summary_path: Path,
    missing_distributions: list[str],
    kept_wheels: list[Path],
    removed_wheels: list[Path],
) -> None:
    lines = ["## LoongSuite PyPI dev bootstrap", ""]
    if missing_distributions:
        lines.append("New PyPI projects to create:")
        lines.extend(f"- `{name}`" for name in missing_distributions)
    else:
        lines.append("No missing PyPI projects were found.")
    lines.extend(
        [
            "",
            f"Kept wheels: {len(kept_wheels)}",
            f"Removed wheels: {len(removed_wheels)}",
        ]
    )
    with summary_path.open("a", encoding="utf-8") as summary:
        summary.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune built LoongSuite wheels to new PyPI project names."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "dist-pypi",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--github-output",
        type=Path,
        default=Path(os.environ["GITHUB_OUTPUT"])
        if "GITHUB_OUTPUT" in os.environ
        else None,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(os.environ["GITHUB_STEP_SUMMARY"])
        if "GITHUB_STEP_SUMMARY" in os.environ
        else None,
    )
    args = parser.parse_args()

    expected_distributions = list_pypi_distribution_names(args.base_dir)
    if not expected_distributions:
        raise RuntimeError(
            "No PyPI distributions were found in the LoongSuite manifest; "
            "refusing to prune wheels."
        )
    missing_distributions = [
        name
        for name in expected_distributions
        if not project_exists_on_pypi(name, timeout=args.timeout)
    ]
    missing_normalized = {
        normalize_distribution_name(name) for name in missing_distributions
    }

    kept_wheels: list[Path] = []
    removed_wheels: list[Path] = []
    for wheel_path in sorted(args.dist_dir.glob("*.whl")):
        wheel_distribution = normalize_distribution_name(
            distribution_name_from_wheel(wheel_path)
        )
        if wheel_distribution in missing_normalized:
            kept_wheels.append(wheel_path)
        else:
            removed_wheels.append(wheel_path)
            if not args.dry_run:
                wheel_path.unlink()

    kept_normalized = {
        normalize_distribution_name(distribution_name_from_wheel(wheel))
        for wheel in kept_wheels
    }
    missing_without_wheel = sorted(missing_normalized - kept_normalized)
    if missing_without_wheel:
        raise RuntimeError(
            "PyPI projects are missing, but no matching wheel was built: "
            + ", ".join(missing_without_wheel)
        )

    print("Missing PyPI projects:")
    for name in missing_distributions or ["(none)"]:
        print(f"  - {name}")
    print(f"Kept wheels: {len(kept_wheels)}")
    print(f"Removed wheels: {len(removed_wheels)}")

    outputs = {
        "has_new_projects": "true" if missing_distributions else "false",
        "missing_distributions": ",".join(missing_distributions),
    }
    if args.github_output:
        write_github_outputs(args.github_output, outputs)
    if args.summary:
        write_step_summary(
            args.summary,
            missing_distributions,
            kept_wheels,
            removed_wheels,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
