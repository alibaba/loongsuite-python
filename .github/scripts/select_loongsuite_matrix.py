#!/usr/bin/env python3
"""Build a dynamic GitHub Actions matrix for selected LoongSuite jobs.

Inputs come from the GitHub Actions environment:
- LOONGSUITE_ALL_JOBS is a JSON list generated into the workflow template.
- LOONGSUITE_FULL is the detector's full output.
- LOONGSUITE_PACKAGES is the detector's pipe-delimited package output.
- GITHUB_OUTPUT receives has_jobs and matrix outputs.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def _parse_packages(package_output: str) -> set[str]:
    return {package for package in package_output.split("|") if package}


def select_jobs(
    all_jobs: list[dict[str, Any]],
    *,
    full: bool,
    package_output: str,
) -> list[dict[str, Any]]:
    # Full is authoritative even when detector fallback emits packages=||.
    if full:
        return all_jobs

    packages = _parse_packages(package_output)
    if not packages:
        return []

    return [
        job
        for job in all_jobs
        if isinstance(job.get("package"), str) and job["package"] in packages
    ]


def _load_all_jobs() -> list[dict[str, Any]]:
    raw_jobs = os.environ.get("LOONGSUITE_ALL_JOBS")
    if raw_jobs is None:
        raise ValueError("LOONGSUITE_ALL_JOBS is required")

    parsed_jobs = json.loads(raw_jobs)
    if not isinstance(parsed_jobs, list):
        raise ValueError(
            "LOONGSUITE_ALL_JOBS must be a JSON list; check generated "
            "job_datas for this template"
        )
    if not parsed_jobs:
        raise ValueError(
            "LOONGSUITE_ALL_JOBS is empty; check generated job_datas for "
            "this template"
        )

    return parsed_jobs


def _write_outputs(outputs: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return

    with open(output_path, "a", encoding="utf-8") as output_file:
        for name, value in outputs.items():
            output_file.write(f"{name}={value}\n")


def main() -> int:
    try:
        all_jobs = _load_all_jobs()
        selected_jobs = select_jobs(
            all_jobs,
            full=os.environ.get("LOONGSUITE_FULL") == "true",
            package_output=os.environ.get("LOONGSUITE_PACKAGES", "||"),
        )
        matrix = json.dumps(
            {"include": selected_jobs},
            separators=(",", ":"),
        )
        outputs = {
            "has_jobs": str(bool(selected_jobs)).lower(),
            "matrix": matrix,
        }
    except Exception as exc:  # noqa: BLE001 - fail the selector job loudly.
        print(
            f"::error::LoongSuite matrix selector failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise

    _write_outputs(outputs)
    print(f"has_jobs={outputs['has_jobs']}")
    print(f"selected_jobs={len(selected_jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
