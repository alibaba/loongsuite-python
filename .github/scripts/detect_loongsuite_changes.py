#!/usr/bin/env python3
"""Detect changed LoongSuite packages for generated GitHub Actions jobs.

Inputs come from the GitHub Actions environment:
- GITHUB_EVENT_NAME and GITHUB_EVENT_PATH describe the current event.
- LOONGSUITE_CHANGED_FILES can inject a newline-separated file list in tests.
- GITHUB_OUTPUT receives full/packages/degraded/reason outputs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

FULL_RUN_LABELS = {"prepare-release", "backport"}
FULL_RUN_FILES = {
    ".github/scripts/detect_loongsuite_changes.py",
    ".github/scripts/select_loongsuite_matrix.py",
    ".github/workflows/generate_workflows_loongsuite.py",
    ".pre-commit-config.yaml",
    ".pylintrc",
    "dev-requirements.txt",
    "eachdist.ini",
    "gen-requirements.txt",
    "pkg-requirements.txt",
    "pyproject.toml",
    "pytest.ini",
    "test-constraints.txt",
    "tox-uv.toml",
    "uv.lock",
}
FULL_RUN_PREFIXES = (
    ".github/scripts/tests/",
    ".github/workflows/generate_workflows_lib/",
    ".github/workflows/loongsuite_",
    ".github/workflows/loongsuite-",
    "loongsuite-distro/",
    "loongsuite-site-bootstrap/",
    "scripts/loongsuite/",
)
TOX_LOONGSUITE_INI_PATH = "tox-loongsuite.ini"
UTIL_GENAI_PREFIX = "util/opentelemetry-util-genai/"
LOONGSUITE_INSTRUMENTATION_PREFIX = "instrumentation-loongsuite/"
BOOTSTRAP_REGISTRY_PREFIX = (
    "loongsuite-distro/src/loongsuite/distro/bootstrap_registry/"
)
DOC_ONLY_SUFFIXES = (".md", ".rst")
REPO_ROOT = Path.cwd()
TOX_LOONGSUITE_INI = REPO_ROOT / TOX_LOONGSUITE_INI_PATH
PACKAGE_MENTION_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<package>loongsuite-instrumentation-[A-Za-z0-9-]+|util-genai)"
    r"(?![A-Za-z0-9-])"
)


def _load_event() -> dict:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return {}

    try:
        with open(event_path, encoding="utf-8") as event_file:
            return json.load(event_file)
    except OSError as exc:
        print(f"Unable to read GitHub event: {exc}", file=sys.stderr)
        return {}


def _run_git_diff(base_ref: str) -> list[str]:
    if not _git_ref_exists(base_ref):
        subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", base_ref],
            check=True,
        )

    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        check=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
    )
    return [
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    ]


def _git_ref_exists(ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", ref],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _pull_request_base_sha(event: dict) -> str:
    pull_request = event.get("pull_request", {})
    base_sha = pull_request.get("base", {}).get("sha")
    if not base_sha:
        raise RuntimeError("pull request base SHA is unavailable")

    return base_sha


def _changed_files(event: dict) -> list[str]:
    if "LOONGSUITE_CHANGED_FILES" in os.environ:
        changed_files = os.environ["LOONGSUITE_CHANGED_FILES"]
        return [
            line.strip() for line in changed_files.splitlines() if line.strip()
        ]

    return _run_git_diff(_pull_request_base_sha(event))


def _has_full_run_label(event: dict) -> bool:
    pull_request = event.get("pull_request", {})
    labels = {
        label.get("name")
        for label in pull_request.get("labels", [])
        if label.get("name")
    }
    return bool(labels & FULL_RUN_LABELS)


def _is_release_pull_request(event: dict) -> bool:
    pull_request = event.get("pull_request", {})
    base_ref = pull_request.get("base", {}).get("ref", "")
    head_ref = pull_request.get("head", {}).get("ref", "")
    return base_ref.startswith("release/") or head_ref.startswith("release/")


def _loongsuite_package_from_path(path: str) -> str | None:
    normalized = path.strip("/")
    if not normalized.startswith(LOONGSUITE_INSTRUMENTATION_PREFIX):
        return None

    package_path = PurePosixPath(normalized)
    if len(package_path.parts) < 2:
        return None

    package = package_path.parts[1]
    if package.startswith("loongsuite-instrumentation-"):
        return package

    return None


def _loongsuite_package_from_bootstrap_registry_path(path: str) -> str | None:
    normalized = path.strip("/")
    if not normalized.startswith(BOOTSTRAP_REGISTRY_PREFIX):
        return None

    relative_path = normalized.removeprefix(BOOTSTRAP_REGISTRY_PREFIX)
    if "/" in relative_path:
        return None

    registry_path = PurePosixPath(relative_path)
    if registry_path.suffix != ".py":
        return None

    module_name = registry_path.stem
    if module_name == "__init__" or module_name.startswith("_"):
        return None

    package = module_name.replace("_", "-")
    if package.startswith("loongsuite-instrumentation-"):
        return package

    return None


def _known_loongsuite_packages() -> set[str]:
    text = TOX_LOONGSUITE_INI.read_text(encoding="utf-8")
    packages = set()
    for line in text.splitlines():
        if line.lstrip().startswith(";"):
            continue
        packages.update(_package_mentions(line))

    return packages


def _is_doc_only_package_path(path: str) -> bool:
    package_path = PurePosixPath(path.strip("/"))
    if len(package_path.parts) <= 2:
        return False

    relative_parts = package_path.parts[2:]
    filename = relative_parts[-1]
    if filename.startswith("CHANGELOG"):
        return True
    if filename.endswith(DOC_ONLY_SUFFIXES):
        return True
    return "docs" in relative_parts


def _requires_full_run(path: str) -> bool:
    normalized = path.strip("/")
    return (
        normalized in FULL_RUN_FILES
        or normalized.startswith(FULL_RUN_PREFIXES)
        or normalized.startswith(UTIL_GENAI_PREFIX)
    )


def _is_generated_loongsuite_workflow(path: str) -> bool:
    normalized = path.strip("/")
    return (
        re.fullmatch(
            r"\.github/workflows/loongsuite_(?:lint|misc|test)_\d+\.yml",
            normalized,
        )
        is not None
    )


def _tox_diff(base_ref: str) -> str:
    if "LOONGSUITE_TOX_DIFF" in os.environ:
        return os.environ["LOONGSUITE_TOX_DIFF"]

    if not _git_ref_exists(base_ref):
        subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", base_ref],
            check=True,
        )

    completed = subprocess.run(
        [
            "git",
            "diff",
            "--unified=0",
            f"{base_ref}...HEAD",
            "--",
            TOX_LOONGSUITE_INI_PATH,
        ],
        check=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def _changed_tox_lines(diff_text: str) -> list[str]:
    lines = []
    for line in diff_text.splitlines():
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue

        content = line[1:].strip()
        if not content or content.startswith(";"):
            continue

        lines.append(content)

    return lines


def _package_mentions(text: str) -> set[str]:
    return {
        match.group("package") for match in PACKAGE_MENTION_RE.finditer(text)
    }


def _packages_from_text(text: str, known_packages: set[str]) -> set[str]:
    return _package_mentions(text) & known_packages


def _packages_from_tox_changes(
    base_ref: str,
    known_packages: set[str],
) -> tuple[set[str], list[str]]:
    packages = set()
    unscoped_lines = []

    for line in _changed_tox_lines(_tox_diff(base_ref)):
        if "util-genai" in _package_mentions(line):
            unscoped_lines.append(line)
            continue

        line_packages = _packages_from_text(line, known_packages)
        if line_packages:
            packages.update(line_packages)
        else:
            unscoped_lines.append(line)

    return packages, unscoped_lines


def _write_outputs(outputs: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return

    with open(output_path, "a", encoding="utf-8") as output_file:
        for name, value in outputs.items():
            output_file.write(f"{name}={value}\n")


def _full_outputs(reason: str, *, degraded: bool = False) -> dict[str, str]:
    return {
        "full": "true",
        "packages": "||",
        "degraded": str(degraded).lower(),
        "reason": reason.replace("\n", " "),
    }


def _package_outputs(packages: set[str], reason: str) -> dict[str, str]:
    package_output = "|" + "|".join(sorted(packages)) + "|"
    return {
        "full": "false",
        "packages": package_output,
        "degraded": "false",
        "reason": reason.replace("\n", " "),
    }


def _detect_outputs() -> dict[str, str]:
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event = _load_event()

    # Push and merge_group events intentionally run the full suite.
    if event_name != "pull_request":
        return _full_outputs("non-pull-request event")

    if _has_full_run_label(event) or _is_release_pull_request(event):
        return _full_outputs("release or backport pull request")

    packages: set[str] = set()
    unknown_packages: set[str] = set()
    tox_changed = False
    changed_files = _changed_files(event)
    if not changed_files:
        return _full_outputs("unable to compute changed files", degraded=True)
    tox_base_sha = None
    if any(
        changed_file.strip("/") == TOX_LOONGSUITE_INI_PATH
        for changed_file in changed_files
    ):
        try:
            tox_base_sha = _pull_request_base_sha(event)
        except RuntimeError as exc:
            return _full_outputs(str(exc), degraded=True)

    known_packages = _known_loongsuite_packages()
    if not known_packages:
        return _full_outputs(
            "unable to determine registered LoongSuite packages",
            degraded=True,
        )

    for changed_file in changed_files:
        normalized = changed_file.strip("/")
        if _is_generated_loongsuite_workflow(normalized):
            continue

        if normalized == TOX_LOONGSUITE_INI_PATH:
            tox_changed = True
            continue

        registry_package = _loongsuite_package_from_bootstrap_registry_path(
            changed_file
        )
        if registry_package:
            if registry_package not in known_packages:
                unknown_packages.add(registry_package)
            else:
                packages.add(registry_package)
            continue

        if _requires_full_run(changed_file):
            return _full_outputs(
                f"shared LoongSuite file changed: {changed_file}"
            )

        package = _loongsuite_package_from_path(changed_file)
        if not package or _is_doc_only_package_path(changed_file):
            continue

        if package not in known_packages:
            unknown_packages.add(package)
        else:
            packages.add(package)

    if tox_changed:
        tox_packages, unscoped_lines = _packages_from_tox_changes(
            tox_base_sha,
            known_packages,
        )
        if unscoped_lines:
            return _full_outputs(
                "shared tox-loongsuite.ini change: " + unscoped_lines[0]
            )
        packages.update(tox_packages)

    if unknown_packages:
        package_list = ", ".join(sorted(unknown_packages))
        print(
            "::error::Changed LoongSuite package is not registered in "
            f"tox-loongsuite.ini: {package_list}",
            file=sys.stderr,
        )
        return _full_outputs(
            f"unknown LoongSuite package changed: {package_list}"
        )

    if packages:
        return _package_outputs(packages, "package-scoped LoongSuite change")

    return _package_outputs(set(), "no LoongSuite package changes")


def _print_outputs(outputs: dict[str, str]) -> None:
    print(f"full={outputs['full']}")
    print(f"packages={outputs['packages']}")
    print(f"degraded={outputs['degraded']}")
    print(f"reason={outputs['reason']}")


def main() -> int:
    try:
        outputs = _detect_outputs()
    except Exception as exc:  # noqa: BLE001 - CI must fall back to full run.
        outputs = _full_outputs(
            f"unexpected detector failure: {type(exc).__name__}: {exc}",
            degraded=True,
        )

    _write_outputs(outputs)
    _print_outputs(outputs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
