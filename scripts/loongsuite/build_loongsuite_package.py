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
LoongSuite Release Build Script

This script supports the following release modes:

1. --build-pypi: Build packages for PyPI publishing
   - loongsuite-otel-util-genai (renamed from opentelemetry-util-genai)
   - loongsuite-distro
   - loongsuite-site-bootstrap
   - loongsuite-instrumentation-* (each package under instrumentation-loongsuite/)

2. --build-github-release: Build packages for GitHub Release (tar.gz)
   - instrumentation-genai/ packages (renamed to loongsuite-*, depends on loongsuite-otel-util-genai)
   - instrumentation-loongsuite/ packages (depends on loongsuite-otel-util-genai)

Version replacement:
- --version: Sets version for all packages being built
- --upstream-version: Sets version for upstream opentelemetry-instrumentation-* packages
  (used in bootstrap_gen.py)

Dependency replacement:
- opentelemetry-util-genai -> loongsuite-otel-util-genai (with ~= version spec)

Package name replacement (for instrumentation-genai/):
- opentelemetry-instrumentation-* -> loongsuite-instrumentation-*
"""

import argparse
import logging
import re
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import tomlkit
from loongsuite_pypi_manifest import (
    PYPI_SKIP_INSTRUMENTATION_LOONGSUITE,
    load_skip_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def depends_on_util_genai(pyproject_path: Path) -> bool:
    """Check if a package depends on opentelemetry-util-genai by reading pyproject.toml."""
    if not pyproject_path.exists():
        return False

    content = pyproject_path.read_text(encoding="utf-8")
    return "opentelemetry-util-genai" in content


def package_release_modifications(
    package_dir: Path, util_dep_spec: str
) -> Dict[str, Any]:
    """Pyproject patches for release builds: util-genai dep + optional genai package rename."""
    modifications: Dict[str, Any] = {}
    pyproject_path = package_dir / "pyproject.toml"

    if depends_on_util_genai(pyproject_path):
        modifications["replace_dependency"] = {
            "old_pattern": "opentelemetry-util-genai",
            "new_value": util_dep_spec,
        }

    if should_rename_package(package_dir):
        pkg_name = package_dir.name
        new_name = pkg_name.replace("opentelemetry-", "loongsuite-")
        modifications["name"] = new_name

    return modifications


def should_rename_package(package_dir: Path) -> bool:
    """
    Determine if a package should be renamed from opentelemetry-* to loongsuite-*.

    Rule: All packages under instrumentation-genai/ with opentelemetry-* prefix
    should be renamed to loongsuite-* prefix.
    """
    # Check if under instrumentation-genai directory
    if "instrumentation-genai" not in str(package_dir):
        return False

    # Check if package name starts with opentelemetry-
    return package_dir.name.startswith("opentelemetry-")


def get_package_name_from_whl(whl_path: Path) -> str:
    """Extract package name from whl filename.

    Wheel filename format: {package}-{version}-{python}-{abi}-{platform}.whl
    Example: loongsuite_instrumentation_openai_v2-0.1.0-py3-none-any.whl

    Note: Package names may contain version-like parts (e.g., 'v2', 'agents-v2')
    that should NOT be treated as version numbers.
    """
    name = whl_path.stem
    parts = name.split("-")
    if len(parts) >= 2:
        package_parts = []
        for part in parts:
            # Check if this looks like a version number:
            # - Starts with digit and contains dot (e.g., "0.1.0", "1.2.3")
            # - Or is a known build tag
            is_version = (
                (part and part[0].isdigit() and "." in part)  # e.g., "0.1.0"
                or part in ("dev", "b0", "b1", "rc0", "rc1")
                or part.startswith("py")  # Python tag: py3, py2
                or part in ("none", "any")  # ABI/platform tags
            )
            if is_version:
                break
            package_parts.append(part)
        return "-".join(package_parts).replace("_", "-")
    return name.replace("_", "-")


@contextmanager
def _patch_pyproject(pyproject_path: Path, modifications: Dict[str, Any]):
    """
    Temporarily patch pyproject.toml using TOML parsing, restore on exit.

    Args:
        pyproject_path: Path to pyproject.toml
        modifications: Dict with optional keys:
            - "name": New package name (str)
            - "replace_dependency": Dict with "old_pattern" and "new_value"
              e.g., {"old_pattern": "opentelemetry-util-genai", "new_value": "loongsuite-otel-util-genai ~= 0.1.0"}
    """
    original_content = pyproject_path.read_text(encoding="utf-8")
    try:
        doc = tomlkit.parse(original_content)

        # Modify package name if specified
        if "name" in modifications:
            doc["project"]["name"] = modifications["name"]

        # Replace dependency if specified
        if "replace_dependency" in modifications:
            old_pattern = modifications["replace_dependency"]["old_pattern"]
            new_value = modifications["replace_dependency"]["new_value"]

            if "dependencies" in doc["project"]:
                deps = doc["project"]["dependencies"]
                new_deps = []
                for dep in deps:
                    # Check if this dependency matches the pattern (package name prefix)
                    # e.g., "opentelemetry-util-genai >= 0.2b0" matches "opentelemetry-util-genai"
                    dep_str = str(dep).strip()
                    dep_name = re.split(r"[<>=~!\s\[]", dep_str)[0].strip()
                    if dep_name == old_pattern:
                        new_deps.append(new_value)
                    else:
                        new_deps.append(dep)
                doc["project"]["dependencies"] = new_deps

        pyproject_path.write_text(tomlkit.dumps(doc), encoding="utf-8")
        yield
    finally:
        pyproject_path.write_text(original_content, encoding="utf-8")


@contextmanager
def _patch_version_py(version_py_path: Path, new_version: str):
    """Temporarily patch version.py, restore on exit."""
    if not version_py_path.exists():
        yield
        return

    content = version_py_path.read_text(encoding="utf-8")
    try:
        patched = re.sub(
            r'__version__\s*=\s*["\'][^"\']*["\']',
            f'__version__ = "{new_version}"',
            content,
        )
        version_py_path.write_text(patched, encoding="utf-8")
        yield
    finally:
        version_py_path.write_text(content, encoding="utf-8")


def find_version_py(package_dir: Path) -> Optional[Path]:
    """Find the version file used by the package build backend."""
    pyproject_path = package_dir / "pyproject.toml"
    if pyproject_path.exists():
        try:
            doc = tomlkit.parse(pyproject_path.read_text(encoding="utf-8"))
            version_config = (
                doc.get("tool", {}).get("hatch", {}).get("version", {})
            )
            version_path = version_config.get("path")
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Failed to read hatch version path from %s: %s",
                pyproject_path,
                exc,
            )
            version_path = None

        if version_path:
            candidate = package_dir / str(version_path)
            if candidate.is_file():
                return candidate

    for version_py in sorted(package_dir.rglob("version.py")):
        if "site-packages" not in str(version_py):
            return version_py
    return None


def build_package(
    package_dir: Path,
    dist_dir: Path,
    existing_whl_files: Set[Path],
) -> List[Path]:
    """Build whl file for a single package"""
    pyproject_toml = package_dir / "pyproject.toml"
    if not pyproject_toml.exists():
        logger.debug(f"Skipping {package_dir}, no pyproject.toml")
        return []

    logger.info(f"Building package: {package_dir}")
    try:
        before_whl_files = set(dist_dir.glob("*.whl"))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(dist_dir),
            ],
            cwd=package_dir,
            check=True,
            capture_output=True,
            text=True,
        )

        after_whl_files = set(dist_dir.glob("*.whl"))
        new_whl_files = [
            f for f in after_whl_files - before_whl_files if f.suffix == ".whl"
        ]

        if not new_whl_files:
            logger.warning(
                f"No new whl files found after building {package_dir}"
            )
            if result.stdout:
                logger.debug(f"stdout: {result.stdout}")
            if result.stderr:
                logger.debug(f"stderr: {result.stderr}")

        return sorted(new_whl_files)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build {package_dir}: {e}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return []


def build_pypi_packages(
    base_dir: Path,
    dist_dir: Path,
    version: str,
    util_genai_version: Optional[str] = None,
    skip_packages: Optional[Set[str]] = None,
) -> List[Path]:
    """
    Build packages for PyPI:
    - loongsuite-otel-util-genai (renamed from opentelemetry-util-genai)
    - loongsuite-distro
    - loongsuite-site-bootstrap
    - each loongsuite-instrumentation-* under instrumentation-loongsuite/
    """
    all_whl_files = []
    existing_whl_files = set(dist_dir.glob("*.whl"))
    skip_packages = skip_packages or set()

    util_ver = util_genai_version or version
    util_dep_spec = f"loongsuite-otel-util-genai ~= {util_ver}"

    # 1. Build util/opentelemetry-util-genai as loongsuite-otel-util-genai
    util_genai_dir = base_dir / "util" / "opentelemetry-util-genai"
    if (
        util_genai_dir.exists()
        and (util_genai_dir / "pyproject.toml").exists()
    ):
        logger.info(
            f"Building loongsuite-otel-util-genai (version {util_ver})..."
        )
        version_py = find_version_py(util_genai_dir)

        modifications = {
            "name": "loongsuite-otel-util-genai",
        }

        with _patch_pyproject(
            util_genai_dir / "pyproject.toml", modifications
        ):
            with (
                _patch_version_py(version_py, util_ver)
                if version_py
                else nullcontext()
            ):
                whl_files = build_package(
                    util_genai_dir, dist_dir, existing_whl_files
                )
                all_whl_files.extend(whl_files)
                existing_whl_files.update(whl_files)

    # 2. Build loongsuite-distro
    distro_dir = base_dir / "loongsuite-distro"
    if distro_dir.exists() and (distro_dir / "pyproject.toml").exists():
        logger.info(f"Building loongsuite-distro (version {version})...")
        version_py = find_version_py(distro_dir)

        with (
            _patch_version_py(version_py, version)
            if version_py
            else nullcontext()
        ):
            whl_files = build_package(distro_dir, dist_dir, existing_whl_files)
            all_whl_files.extend(whl_files)
            existing_whl_files.update(whl_files)

    # 3. Build loongsuite-site-bootstrap
    site_bootstrap_dir = base_dir / "loongsuite-site-bootstrap"
    if (
        site_bootstrap_dir.exists()
        and (site_bootstrap_dir / "pyproject.toml").exists()
    ):
        logger.info(
            f"Building loongsuite-site-bootstrap (version {version})..."
        )
        version_py = find_version_py(site_bootstrap_dir)
        modifications = {
            "replace_dependency": {
                "old_pattern": "loongsuite-distro",
                "new_value": f"loongsuite-distro ~= {version}",
            },
        }
        with _patch_pyproject(
            site_bootstrap_dir / "pyproject.toml", modifications
        ):
            with (
                _patch_version_py(version_py, version)
                if version_py
                else nullcontext()
            ):
                whl_files = build_package(
                    site_bootstrap_dir, dist_dir, existing_whl_files
                )
                all_whl_files.extend(whl_files)
                existing_whl_files.update(whl_files)

    # 4. Build instrumentation-loongsuite/* (loongsuite-instrumentation-* on PyPI)
    instrumentation_loongsuite_dir = base_dir / "instrumentation-loongsuite"
    if instrumentation_loongsuite_dir.exists():
        logger.info("Building instrumentation-loongsuite packages for PyPI...")
        for package_dir in sorted(instrumentation_loongsuite_dir.iterdir()):
            if (
                not package_dir.is_dir()
                or not (package_dir / "pyproject.toml").exists()
            ):
                continue

            pkg_name = package_dir.name
            if pkg_name in skip_packages:
                logger.info(f"Skipping {pkg_name} (in skip list)")
                continue
            if pkg_name in PYPI_SKIP_INSTRUMENTATION_LOONGSUITE:
                logger.info(
                    f"Skipping {pkg_name} for PyPI (FIXME: enable after sufficient testing)"
                )
                continue

            modifications = package_release_modifications(
                package_dir, util_dep_spec
            )
            version_py = find_version_py(package_dir)

            logger.info(f"Building {pkg_name} for PyPI (version {version})...")
            with (
                _patch_pyproject(package_dir / "pyproject.toml", modifications)
                if modifications
                else nullcontext()
            ):
                with (
                    _patch_version_py(version_py, version)
                    if version_py
                    else nullcontext()
                ):
                    whl_files = build_package(
                        package_dir, dist_dir, existing_whl_files
                    )
                    all_whl_files.extend(whl_files)
                    existing_whl_files.update(whl_files)

    return all_whl_files


def build_github_release_packages(
    base_dir: Path,
    dist_dir: Path,
    version: str,
    util_genai_version: Optional[str] = None,
    skip_packages: Optional[Set[str]] = None,
) -> List[Path]:
    """
    Build packages for GitHub Release (tar.gz):
    - instrumentation-genai/ (renamed to loongsuite-*, depends on loongsuite-otel-util-genai)
    - instrumentation-loongsuite/ (depends on loongsuite-otel-util-genai)
    """
    all_whl_files = []
    existing_whl_files = set(dist_dir.glob("*.whl"))
    skip_packages = skip_packages or set()

    util_ver = util_genai_version or version
    util_dep_spec = f"loongsuite-otel-util-genai ~= {util_ver}"

    # 1. Build instrumentation-genai/ packages
    instrumentation_genai_dir = base_dir / "instrumentation-genai"
    if instrumentation_genai_dir.exists():
        logger.info("Building packages under instrumentation-genai/...")
        for package_dir in sorted(instrumentation_genai_dir.iterdir()):
            if (
                not package_dir.is_dir()
                or not (package_dir / "pyproject.toml").exists()
            ):
                continue

            pkg_name = package_dir.name
            if pkg_name in skip_packages:
                logger.info(f"Skipping {pkg_name} (in skip list)")
                continue

            modifications = package_release_modifications(
                package_dir, util_dep_spec
            )
            version_py = find_version_py(package_dir)

            logger.info(f"Building {pkg_name} (version {version})...")
            with (
                _patch_pyproject(package_dir / "pyproject.toml", modifications)
                if modifications
                else nullcontext()
            ):
                with (
                    _patch_version_py(version_py, version)
                    if version_py
                    else nullcontext()
                ):
                    whl_files = build_package(
                        package_dir, dist_dir, existing_whl_files
                    )
                    all_whl_files.extend(whl_files)
                    existing_whl_files.update(whl_files)

    # 2. Build instrumentation-loongsuite/ packages
    instrumentation_loongsuite_dir = base_dir / "instrumentation-loongsuite"
    if instrumentation_loongsuite_dir.exists():
        logger.info("Building packages under instrumentation-loongsuite/...")
        for package_dir in sorted(instrumentation_loongsuite_dir.iterdir()):
            if (
                not package_dir.is_dir()
                or not (package_dir / "pyproject.toml").exists()
            ):
                continue

            pkg_name = package_dir.name
            if pkg_name in skip_packages:
                logger.info(f"Skipping {pkg_name} (in skip list)")
                continue

            modifications = package_release_modifications(
                package_dir, util_dep_spec
            )
            version_py = find_version_py(package_dir)

            logger.info(f"Building {pkg_name} (version {version})...")
            with (
                _patch_pyproject(package_dir / "pyproject.toml", modifications)
                if modifications
                else nullcontext()
            ):
                with (
                    _patch_version_py(version_py, version)
                    if version_py
                    else nullcontext()
                ):
                    whl_files = build_package(
                        package_dir, dist_dir, existing_whl_files
                    )
                    all_whl_files.extend(whl_files)
                    existing_whl_files.update(whl_files)

    return all_whl_files


def _filter_and_dedupe_whl_files(
    all_whl_files: List[Path],
    skip_packages: Set[str],
) -> List[Path]:
    """Filter skip list and deduplicate whl files."""
    filtered_whl_files = []
    skipped_count = 0
    seen_packages = {}

    for whl_file in all_whl_files:
        package_name = get_package_name_from_whl(whl_file)

        if package_name in skip_packages:
            logger.info(f"Skipping package: {package_name} (in skip list)")
            skipped_count += 1
            whl_file.unlink()
            continue

        if package_name in seen_packages:
            existing_file = seen_packages[package_name]
            if whl_file.stat().st_mtime > existing_file.stat().st_mtime:
                logger.debug(f"Replacing duplicate {package_name}")
                existing_file.unlink()
                seen_packages[package_name] = whl_file
                filtered_whl_files.remove(existing_file)
                filtered_whl_files.append(whl_file)
            else:
                whl_file.unlink()
        else:
            seen_packages[package_name] = whl_file
            filtered_whl_files.append(whl_file)

    logger.info(f"Built {len(all_whl_files)} whl files")
    logger.info(f"Skipped {skipped_count} packages")
    logger.info(f"Final: {len(filtered_whl_files)} whl files")

    return filtered_whl_files


def create_tar_archive(whl_files: List[Path], output_path: Path):
    """Package all whl files into tar.gz"""
    logger.info(f"Creating tar archive: {output_path}")

    with tarfile.open(output_path, "w:gz") as tar:
        for whl_file in sorted(whl_files):
            tar.add(whl_file, arcname=whl_file.name)
            logger.debug(f"Added: {whl_file.name}")

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Created: {output_path} ({size_mb:.2f} MB)")


# Python 3.9 compatibility: nullcontext
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext():
        yield


def main():
    parser = argparse.ArgumentParser(
        description="LoongSuite Release Build Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build for PyPI (util-genai, distro, instrumentation-loongsuite wheels)
  python build_loongsuite_package.py --build-pypi --version 0.1.0

  # Build for GitHub Release (instrumentation packages)
  python build_loongsuite_package.py --build-github-release --version 0.1.0

  # Build both
  python build_loongsuite_package.py --build-pypi --build-github-release --version 0.1.0
        """,
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent,
        help="Project root directory",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=None,
        help="Build output directory (default: base-dir/dist)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "loongsuite-build-config.json",
        help="Config file for skip packages",
    )

    # Build mode
    parser.add_argument(
        "--build-pypi",
        action="store_true",
        help=(
            "Build packages for PyPI (loongsuite-otel-util-genai, loongsuite-distro, "
            "loongsuite-site-bootstrap, instrumentation-loongsuite/*)"
        ),
    )
    parser.add_argument(
        "--build-github-release",
        action="store_true",
        help="Build packages for GitHub Release (instrumentation-genai, instrumentation-loongsuite)",
    )

    # Legacy mode (for backward compatibility)
    parser.add_argument(
        "--loongsuite-release",
        action="store_true",
        help="(Legacy) Same as --build-github-release",
    )

    # Version settings
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version for all packages",
    )
    parser.add_argument(
        "--util-genai-version",
        type=str,
        default=None,
        help="Version for loongsuite-otel-util-genai (default: same as --version)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output tar.gz path (for GitHub Release)",
    )

    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    if args.dist_dir is None:
        dist_dir = base_dir / "dist"
    elif args.dist_dir.is_absolute():
        dist_dir = args.dist_dir
    else:
        dist_dir = base_dir / args.dist_dir
    dist_dir.mkdir(parents=True, exist_ok=True)

    # Clean old whl files
    logger.info(f"Cleaning old build files: {dist_dir}")
    for old_file in dist_dir.glob("*.whl"):
        old_file.unlink()

    skip_packages = load_skip_config(args.config)

    # Handle legacy mode
    if args.loongsuite_release:
        args.build_github_release = True

    if not args.build_pypi and not args.build_github_release:
        parser.error(
            "Must specify at least one of --build-pypi or --build-github-release"
        )

    pypi_whl_files = []
    github_whl_files = []

    # Build PyPI packages
    if args.build_pypi:
        logger.info("=" * 50)
        logger.info("Building PyPI packages...")
        logger.info("=" * 50)
        pypi_whl_files = build_pypi_packages(
            base_dir,
            dist_dir,
            args.version,
            args.util_genai_version,
            skip_packages,
        )
        logger.info(f"PyPI packages built: {len(pypi_whl_files)}")
        for whl in pypi_whl_files:
            logger.info(f"  - {whl.name}")

    # Build GitHub Release packages
    if args.build_github_release:
        logger.info("=" * 50)
        logger.info("Building GitHub Release packages...")
        logger.info("=" * 50)
        github_whl_files = build_github_release_packages(
            base_dir,
            dist_dir,
            args.version,
            args.util_genai_version,
            skip_packages,
        )

        github_whl_files = _filter_and_dedupe_whl_files(
            github_whl_files, skip_packages
        )

        if github_whl_files:
            output_path = args.output or (
                dist_dir / f"loongsuite-python-{args.version}.tar.gz"
            )
            create_tar_archive(github_whl_files, output_path)
            logger.info(f"GitHub Release tar: {output_path}")

    logger.info("=" * 50)
    logger.info("Build completed!")
    logger.info("=" * 50)

    if pypi_whl_files:
        logger.info("PyPI packages (upload with twine):")
        for whl in pypi_whl_files:
            logger.info(f"  {whl}")

    if github_whl_files:
        logger.info(
            f"GitHub Release tar ready: {dist_dir}/loongsuite-python-{args.version}.tar.gz"
        )


if __name__ == "__main__":
    main()
