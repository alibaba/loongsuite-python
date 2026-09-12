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

"""
Collect, archive, and bump LoongSuite changelogs / versions.

Modes:
  --collect          Gather all Unreleased sections and emit a release-notes markdown file
                     (includes a PyPI distribution list from loongsuite_pypi_manifest).
  --archive          Replace Unreleased headers with a versioned header in-place
                     (empty Unreleased bodies get a one-line English placeholder).
  --bump-dev            Bump instrumentation-loongsuite, loongsuite-distro, and loongsuite-site-bootstrap versions to the next dev version.
  --pin-release-versions  Pin __version__ to the release version for PyPI packages (release branch; includes util-genai).
  --rename-packages  Rename opentelemetry-util-genai to loongsuite-otel-util-genai in pyproject.toml files.

Changelog sources (in order):
  1. CHANGELOG-loongsuite.md              (root, label: loongsuite)
  2. util/opentelemetry-util-genai/CHANGELOG-loongsuite.md  (label: loongsuite-otel-util-genai)
  3. instrumentation-loongsuite/*/CHANGELOG.md              (per-package)

Usage:
  python scripts/loongsuite/collect_loongsuite_changelog.py --collect \\
      --version 0.1.0 --upstream-version 0.60b1 --output dist/release-notes.md

  python scripts/loongsuite/collect_loongsuite_changelog.py --archive \\
      --version 0.1.0

  python scripts/loongsuite/collect_loongsuite_changelog.py --bump-dev \\
      --version 0.1.0

  python scripts/loongsuite/collect_loongsuite_changelog.py --pin-release-versions \\
      --version 0.1.0
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

UNRELEASED_RE = re.compile(r"^##\s+\[?Unreleased\]?\s*$", re.IGNORECASE)
NEXT_SECTION_RE = re.compile(r"^##\s+")

# Inserted under a new version heading when --archive finds an empty Unreleased body.
NO_CHANGELOG_ENTRIES_LINE = "There are no changelog entries for this release."


def _unreleased_block_is_empty(
    lines: List[str], unreleased_line_index: int
) -> bool:
    """True if the Unreleased section has no non-whitespace body (same bounds as _extract_unreleased)."""
    start = unreleased_line_index + 1
    end = len(lines)
    for j in range(start, len(lines)):
        if NEXT_SECTION_RE.match(lines[j]):
            end = j
            break
    return "\n".join(lines[start:end]).strip() == ""


def _changelog_sources(repo: Path) -> List[Tuple[str, Path]]:
    """Return (label, path) pairs for all changelog sources."""
    sources: List[Tuple[str, Path]] = []

    root_cl = repo / "CHANGELOG-loongsuite.md"
    if root_cl.exists():
        sources.append(("loongsuite", root_cl))

    util_cl = (
        repo / "util" / "opentelemetry-util-genai" / "CHANGELOG-loongsuite.md"
    )
    if util_cl.exists():
        sources.append(("loongsuite-otel-util-genai", util_cl))

    inst_dir = repo / "instrumentation-loongsuite"
    if inst_dir.is_dir():
        for pkg_dir in sorted(inst_dir.iterdir()):
            cl = pkg_dir / "CHANGELOG.md"
            if cl.exists():
                sources.append((pkg_dir.name, cl))

    return sources


def _extract_unreleased(path: Path) -> Optional[str]:
    """Extract the content between the Unreleased header and the next ## header."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if UNRELEASED_RE.match(line):
            start = i + 1
            break

    if start is None:
        return None

    end = len(lines)
    for i in range(start, len(lines)):
        if NEXT_SECTION_RE.match(lines[i]):
            end = i
            break

    # Also match top-level `# Added` etc. that appear after Unreleased (formatting bug in root changelog)
    content_lines = []
    for line in lines[start:end]:
        # Normalise stray top-level `# Foo` to `### Foo` inside an Unreleased block
        if re.match(r"^#\s+\w", line) and not re.match(r"^##", line):
            line = "##" + line  # `# Added` -> `### Added`
        content_lines.append(line)

    content = "\n".join(content_lines).strip()
    return content if content else None


def _collapse_link_linebreaks(text: str) -> str:
    r"""Join lines where a link reference like ``([#N](url))`` is on its own indented line."""
    return re.sub(r"\n[ \t]+(\(\[#)", r" \1", text)


def _list_pypi_distribution_names(repo: Path) -> List[str]:
    """Same inclusion rules as ``build_pypi_packages`` (see ``loongsuite_pypi_manifest``)."""
    script_dir = repo / "scripts" / "loongsuite"
    if not (script_dir / "loongsuite_pypi_manifest.py").is_file():
        return []
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import loongsuite_pypi_manifest as lpm  # noqa: PLC0415

    return lpm.list_pypi_distribution_names(repo)


def collect(
    version: str, upstream_version: str, output: Path, repo: Path
) -> None:
    """Collect all Unreleased sections into a single release-notes file."""
    parts: List[str] = []
    parts.append(f"# loongsuite-python {version}\n")
    parts.append("## Installation\n")
    parts.append("```bash")
    parts.append(f"pip install loongsuite-distro=={version}")
    parts.append(f"loongsuite-bootstrap -a install --version {version}")
    parts.append("```\n")
    parts.append("## Package Versions\n")
    parts.append(f"- loongsuite-* packages: {version}")
    parts.append(f"- opentelemetry-* packages: {upstream_version}\n")
    parts.append("## PyPI packages\n")
    parts.append(
        "The following distributions are built and uploaded to PyPI for this release:\n"
    )
    pypi_dists = _list_pypi_distribution_names(repo)
    if pypi_dists:
        for dist_name in pypi_dists:
            parts.append(f"- `{dist_name}`")
    else:
        parts.append(
            "- _(Could not resolve the list; ensure "
            "`scripts/loongsuite/build_loongsuite_package.py` is present.)_"
        )
    parts.append("")
    parts.append("---\n")

    found_any = False
    first = True
    for label, path in _changelog_sources(repo):
        content = _extract_unreleased(path)
        if content:
            found_any = True
            content = _collapse_link_linebreaks(content)
            if not first:
                parts.append("---\n")
            first = False
            parts.append(f"## {label}\n")
            parts.append(content)
            parts.append("")

    if not found_any:
        parts.append("No unreleased changes found.\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Release notes written to {output}")


def archive(version: str, repo: Path, date_str: Optional[str] = None) -> None:
    """Archive Unreleased sections in-place: insert a versioned header below Unreleased.

    When the Unreleased body is empty (same criterion as _extract_unreleased), inserts
    NO_CHANGELOG_ENTRIES_LINE under the new version heading, with blank lines separating
    it from the version title and the following section.
    """
    if date_str is None:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    version_header = f"## Version {version} ({date_str})"

    for label, path in _changelog_sources(repo):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        new_lines: List[str] = []
        found = False

        i = 0
        while i < len(lines):
            if not found and UNRELEASED_RE.match(lines[i]):
                found = True
                original_header = lines[i]
                new_lines.append(original_header)
                new_lines.append("")
                new_lines.append(version_header)
                new_lines.append("")

                if _unreleased_block_is_empty(lines, i):
                    new_lines.append(NO_CHANGELOG_ENTRIES_LINE)
                    new_lines.append("")

                # Skip blank lines immediately after the old Unreleased header
                i += 1
                while i < len(lines) and lines[i].strip() == "":
                    i += 1
                continue

            new_lines.append(lines[i])
            i += 1

        if found:
            # Ensure file ends with newline
            result = "\n".join(new_lines)
            if not result.endswith("\n"):
                result += "\n"
            path.write_text(result, encoding="utf-8")
            print(f"Archived {label}: {path}")
        else:
            print(f"No Unreleased section in {label}: {path} (skipped)")


# Opening/closing quote must match (\2); optional trailing comment after the string literal.
VERSION_RE = re.compile(
    r'^(__version__\s*=\s*(["\']))([^\n]*?)\2(\s*(?:#.*)?)$',
    re.MULTILINE,
)


def _managed_loongsuite_version_files(
    repo: Path, *, include_util_genai: bool
) -> List[Path]:
    """Paths to version.py files versioned with the LoongSuite release train."""
    version_files: List[Path] = []

    inst_dir = repo / "instrumentation-loongsuite"
    if inst_dir.is_dir():
        version_files.extend(sorted(inst_dir.rglob("version.py")))

    distro_version_py = (
        repo
        / "loongsuite-distro"
        / "src"
        / "loongsuite"
        / "distro"
        / "version.py"
    )
    if distro_version_py.is_file():
        version_files.append(distro_version_py)

    site_version_py = (
        repo
        / "loongsuite-site-bootstrap"
        / "src"
        / "loongsuite_site_bootstrap"
        / "version.py"
    )
    if site_version_py.is_file():
        version_files.append(site_version_py)

    if include_util_genai:
        util_version_py = (
            repo
            / "util"
            / "opentelemetry-util-genai"
            / "src"
            / "opentelemetry"
            / "util"
            / "genai"
            / "version.py"
        )
        if util_version_py.is_file():
            version_files.append(util_version_py)

    return sorted(version_files, key=lambda p: str(p))


def _next_dev_version(released_version: str) -> str:
    """Compute the next development version by bumping the minor segment.

    Examples: "0.1.0" -> "0.2.0.dev", "1.3.2" -> "1.4.0.dev"
    """
    parts = released_version.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Cannot compute next dev version from '{released_version}'"
        )
    major = int(parts[0])
    minor = int(parts[1])
    return f"{major}.{minor + 1}.0.dev"


def pin_release_versions(released_version: str, repo: Path) -> None:
    """Set managed packages' __version__ to the release version (before PyPI publish)."""
    version_files = _managed_loongsuite_version_files(
        repo, include_util_genai=True
    )
    if not version_files:
        print("WARNING: no version.py files to pin for release")
        return

    for vf in version_files:
        text = vf.read_text(encoding="utf-8")
        m = VERSION_RE.search(text)
        if m:
            new_text = VERSION_RE.sub(rf"\g<1>{released_version}\2\4", text)
            vf.write_text(new_text, encoding="utf-8")
            print(
                f'Pinned {vf.relative_to(repo)} -> __version__ = "{released_version}"'
            )
        else:
            print(f"WARNING: no __version__ found in {vf.relative_to(repo)}")


def bump_dev(
    released_version: str, repo: Path, next_version: Optional[str] = None
) -> None:
    """Bump managed packages to the next dev version."""
    next_ver = next_version or _next_dev_version(released_version)
    version_files = _managed_loongsuite_version_files(
        repo, include_util_genai=False
    )

    if not version_files:
        print("WARNING: no version.py files to bump")
        return

    for vf in sorted(version_files):
        text = vf.read_text(encoding="utf-8")
        m = VERSION_RE.search(text)
        if m:
            new_text = VERSION_RE.sub(rf"\g<1>{next_ver}\2\4", text)
            vf.write_text(new_text, encoding="utf-8")
            print(
                f'Bumped {vf.relative_to(repo)}: {m.group(0).strip()} -> __version__ = "{next_ver}"'
            )
        else:
            print(f"WARNING: no __version__ found in {vf.relative_to(repo)}")


def rename_packages(version: str, repo: Path) -> None:
    """Permanently rename opentelemetry-util-genai to loongsuite-otel-util-genai in pyproject.toml files.

    This makes the release branch a self-contained snapshot where package names
    and dependencies already reflect the published names.
    """
    try:
        import tomlkit  # noqa: PLC0415
    except ImportError:
        print(
            "ERROR: tomlkit is required for --rename-packages. Install with: pip install tomlkit"
        )
        sys.exit(1)

    util_dep_spec = f"loongsuite-otel-util-genai ~= {version}"

    # 1. Rename util/opentelemetry-util-genai itself
    util_pyproject = (
        repo / "util" / "opentelemetry-util-genai" / "pyproject.toml"
    )
    if util_pyproject.exists():
        doc = tomlkit.parse(util_pyproject.read_text(encoding="utf-8"))
        old_name = doc["project"]["name"]
        doc["project"]["name"] = "loongsuite-otel-util-genai"
        util_pyproject.write_text(tomlkit.dumps(doc), encoding="utf-8")
        print(
            f"Renamed {util_pyproject.relative_to(repo)}: {old_name} -> loongsuite-otel-util-genai"
        )
    else:
        print(f"WARNING: {util_pyproject} not found")

    # 2. Replace dependency in instrumentation-loongsuite and instrumentation-genai
    for search_dir in ("instrumentation-loongsuite", "instrumentation-genai"):
        inst_dir = repo / search_dir
        if not inst_dir.is_dir():
            continue
        for pyproject in sorted(inst_dir.rglob("pyproject.toml")):
            text = pyproject.read_text(encoding="utf-8")
            if "opentelemetry-util-genai" not in text:
                continue
            doc = tomlkit.parse(text)
            deps = doc.get("project", {}).get("dependencies", [])
            changed = False
            for i, dep in enumerate(deps):
                dep_name = re.split(r"[<>=~!\s\[]", str(dep).strip())[
                    0
                ].strip()
                if dep_name == "opentelemetry-util-genai":
                    deps[i] = util_dep_spec
                    changed = True
            if changed:
                pyproject.write_text(tomlkit.dumps(doc), encoding="utf-8")
                print(f"Updated dependency in {pyproject.relative_to(repo)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect/archive LoongSuite changelogs"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--collect",
        action="store_true",
        help="Collect Unreleased into release notes",
    )
    group.add_argument(
        "--archive",
        action="store_true",
        help="Archive Unreleased to versioned header",
    )
    group.add_argument(
        "--bump-dev",
        action="store_true",
        help=(
            "Bump instrumentation-loongsuite + loongsuite-distro + "
            "loongsuite-site-bootstrap to next dev"
        ),
    )
    group.add_argument(
        "--pin-release-versions",
        action="store_true",
        help=(
            "Pin __version__ to release version for PyPI packages (util-genai, "
            "distro, site-bootstrap, instrumentation-loongsuite)"
        ),
    )
    parser.add_argument(
        "--rename-packages",
        action="store_true",
        default=True,
        help="Rename opentelemetry-util-genai to loongsuite-otel-util-genai (default, always runs unless other mode specified)",
    )

    parser.add_argument(
        "--version", required=True, help="LoongSuite version (e.g. 0.1.0)"
    )
    parser.add_argument(
        "--upstream-version",
        default="",
        help="Upstream OTel version (for --collect header)",
    )
    parser.add_argument(
        "--output",
        default="dist/release-notes.md",
        help="Output file for --collect",
    )
    parser.add_argument(
        "--repo-root", default=str(REPO_ROOT), help="Repository root"
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Release date (YYYY-MM-DD), default: today",
    )
    parser.add_argument(
        "--next-dev-version",
        default=None,
        help="Override next dev version (default: auto-computed)",
    )

    args = parser.parse_args()
    repo = Path(args.repo_root)

    if args.collect:
        if not args.upstream_version:
            parser.error("--upstream-version is required for --collect")
        collect(args.version, args.upstream_version, Path(args.output), repo)
    elif args.archive:
        archive(args.version, repo, args.date)
    elif args.bump_dev:
        bump_dev(args.version, repo, args.next_dev_version)
    elif args.pin_release_versions:
        pin_release_versions(args.version, repo)
    else:
        rename_packages(args.version, repo)


if __name__ == "__main__":
    main()
