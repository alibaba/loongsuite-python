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
LoongSuite Bootstrap Tool

Two-phase installation strategy:
1. Install loongsuite-* packages from GitHub Release tar.gz (GenAI instrumentations)
2. Install opentelemetry-* packages from PyPI (standard instrumentations)

The installation source is determined by package name prefix:
- loongsuite-* -> GitHub Release tar.gz
- opentelemetry-* -> PyPI

loongsuite-otel-util-genai is installed from PyPI as a base dependency.
"""

import argparse
import json as json_lib
import logging
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple, Union

from loongsuite.distro.bootstrap_gen import (
    default_instrumentations as gen_default_instrumentations,
)
from loongsuite.distro.bootstrap_gen import libraries as gen_libraries
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

logger = logging.getLogger(__name__)

# Base dependency packages installed from PyPI
# loongsuite-otel-util-genai is published to PyPI and required by GenAI instrumentations
BASE_DEPENDENCIES_PYPI = {
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-instrumentation",
    "loongsuite-otel-util-genai",
    "opentelemetry-semantic-conventions",
}

# Packages to exclude from uninstallation
UNINSTALL_EXCLUDED_PACKAGES = {
    "loongsuite-distro",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-instrumentation",
}


def normalize_package_name(package_name: str) -> str:
    """
    Normalize package name by converting underscores to hyphens.

    Package names in PyPI use hyphens, but wheel filenames may use underscores.
    This function ensures consistent package name format.

    Args:
        package_name: Package name (may contain underscores or hyphens)

    Returns:
        Normalized package name with hyphens
    """
    return package_name.replace("_", "-")


def get_package_name_variants(package_name: str) -> List[str]:
    """
    Get all possible variants of a package name for lookup.

    This is useful when checking if a package is installed, as package names
    may be stored with either underscores or hyphens.

    Args:
        package_name: Package name

    Returns:
        List of package name variants to try
    """
    variants = [package_name]
    normalized = normalize_package_name(package_name)
    if normalized != package_name:
        variants.append(normalized)
    # Also try reverse (hyphens to underscores) for completeness
    reverse = package_name.replace("-", "_")
    if reverse != package_name and reverse not in variants:
        variants.append(reverse)
    return variants


def extract_package_name_from_requirement(req_str: str) -> str:
    """
    Extract package name from a requirement string.

    Examples:
        "redis >= 2.6" -> "redis"
        "opentelemetry-instrumentation==0.60b0" -> "opentelemetry-instrumentation"
        "package-name~=1.0" -> "package-name"

    Args:
        req_str: Requirement string

    Returns:
        Package name extracted from requirement
    """
    try:
        return Requirement(req_str).name
    except Exception:
        # Fallback: manual parsing if Requirement parsing fails
        for op in ["==", ">=", "<=", "~=", "!=", ">", "<"]:
            if op in req_str:
                return req_str.split(op)[0].strip()
        return req_str.strip()


def package_names_match(name1: str, name2: str) -> bool:
    """
    Check if two package names match (considering normalization).

    Args:
        name1: First package name
        name2: Second package name

    Returns:
        True if names match (after normalization), False otherwise
    """
    normalized1 = normalize_package_name(name1)
    normalized2 = normalize_package_name(name2)
    return (
        normalized1 == normalized2
        or name1 == name2
        or normalized1 == name2
        or name1 == normalized2
    )


def load_list_file(file_path: Path) -> Set[str]:
    """Load list from file (one package name per line)"""
    if not file_path.exists():
        return set()

    packages = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                packages.add(line)

    return packages


def get_package_name_from_whl(whl_path: Path) -> str:
    """
    Extract package name from whl filename

    Wheel filename format: {package_name}-{version}-{python_tag}-{abi_tag}-{platform_tag}.whl
    Example: loongsuite_instrumentation_mem0-0.1.0-py3-none-any.whl

    Returns normalized package name with hyphens (e.g., "loongsuite-instrumentation-mem0")
    """
    name = whl_path.stem  # Remove .whl extension
    parts = name.split("-")

    if len(parts) < 2:
        # If no hyphens, return as-is (shouldn't happen for valid wheels)
        return name.replace("_", "-")

    package_parts = []
    for part in parts:
        # Check if this part looks like a version number
        # Version numbers typically:
        # - Start with a digit
        # - Contain dots (e.g., "0.1.0", "1.2.3")
        # - Or are build tags like "dev", "b0", etc.
        # - Or are Python/ABI/platform tags

        # Check for version-like patterns: starts with digit and contains dot, or is a known tag
        is_version_like = (
            (
                part and part[0].isdigit() and "." in part
            )  # e.g., "0.1.0", "1.2.3"
            or part in ("dev", "b0", "b1", "rc0", "rc1")  # Build tags
            or part.startswith("py")  # Python tags: "py3", "py2", "py"
            or part in ("none", "any")  # ABI/platform tags
        )

        if is_version_like:
            break

        package_parts.append(part)

    if not package_parts:
        # Fallback: if we couldn't extract, use first part
        result = parts[0] if parts else name
    else:
        # Join with hyphens
        result = "-".join(package_parts)

    # Normalize: convert underscores to hyphens for package name consistency
    # (wheel filenames may use underscores, but package names use hyphens)
    result = result.replace("_", "-")
    return result


def get_metadata_from_whl(whl_path: Path) -> Optional[dict[str, Any]]:
    """
    Extract metadata from whl file

    Args:
        whl_path: Path to whl file

    Returns:
        Dictionary with metadata fields, or None if not found
    """
    try:
        with zipfile.ZipFile(whl_path, "r") as whl_zip:
            # Look for METADATA file in the wheel
            metadata_path = None
            for name in whl_zip.namelist():
                if name.endswith("/METADATA") or name == "METADATA":
                    metadata_path = name
                    break

            if not metadata_path:
                return None

            metadata = {}
            current_field = None
            # Read METADATA file
            with whl_zip.open(metadata_path) as metadata_file:
                for line in metadata_file:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        current_field = None
                        continue

                    # Check for continuation line
                    if line_str.startswith(" ") or line_str.startswith("\t"):
                        if current_field and current_field in metadata:
                            if isinstance(metadata[current_field], list):
                                if metadata[current_field]:
                                    metadata[current_field][-1] += (
                                        " " + line_str.strip()
                                    )
                            else:
                                metadata[current_field] += (
                                    " " + line_str.strip()
                                )
                        continue

                    # Parse field name and value
                    if ":" in line_str:
                        field_name, field_value = line_str.split(":", 1)
                        field_name = field_name.strip()
                        field_value = field_value.strip()
                        current_field = field_name

                        if field_name == "Requires-Python":
                            metadata["requires_python"] = field_value
                        elif field_name == "Requires-Dist":
                            if "requires_dist" not in metadata:
                                metadata["requires_dist"] = []
                            metadata["requires_dist"].append(field_value)
                        elif field_name == "Provides-Extra":
                            if "provides_extra" not in metadata:
                                metadata["provides_extra"] = []
                            metadata["provides_extra"].append(field_value)

            return metadata if metadata else None
    except Exception:
        pass

    return None


def get_python_requirement_from_whl(whl_path: Path) -> Optional[str]:
    """
    Extract Python version requirement from whl file metadata

    Args:
        whl_path: Path to whl file

    Returns:
        Python version requirement string (e.g., ">=3.10, <=3.13") or None if not found
    """
    metadata = get_metadata_from_whl(whl_path)
    return metadata.get("requires_python") if metadata else None


def _try_get_package_version(package_name: str) -> Optional[str]:
    """
    Try to get version of a package using pip show.

    Args:
        package_name: Package name to check

    Returns:
        Version string if found, None otherwise
    """
    cmd = [sys.executable, "-m", "pip", "show", package_name]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.debug(f"Failed to get version for {package_name}: {e}")
    except Exception as e:
        logger.warning(
            f"Unexpected error getting version for {package_name}: {e}"
        )
    return None


def get_installed_package_version(package_name: str) -> Optional[str]:
    """
    Get installed version of a package.

    Tries multiple name variants (with underscores/hyphens) to handle
    different naming conventions.

    Args:
        package_name: Package name (may contain hyphens or underscores)

    Returns:
        Installed version string, or None if not installed
    """
    variants = get_package_name_variants(package_name)
    for variant in variants:
        version = _try_get_package_version(variant)
        if version:
            return version
    return None


def _is_library_installed(req_str: str) -> bool:
    """
    Check if a library is installed and version satisfies requirement.

    Similar to opentelemetry-bootstrap's _is_installed function.

    Args:
        req_str: Requirement string (e.g., "redis >= 2.6")

    Returns:
        True if library is installed and version satisfies requirement, False otherwise
    """
    try:
        req = Requirement(req_str)
        package_name = req.name

        # get_installed_package_version already tries multiple variants
        dist_version = get_installed_package_version(package_name)

        if dist_version is None:
            return False

        # Check if installed version satisfies requirement
        return req.specifier.contains(dist_version)
    except Exception as e:
        logger.debug(
            f"Failed to check if library is installed for {req_str}: {e}"
        )
        return False


def _is_instrumentation_in_bootstrap_gen(package_name: str) -> bool:
    """
    Check if a package is an instrumentation listed in bootstrap_gen.py.

    Args:
        package_name: Package name to check

    Returns:
        True if the package is in bootstrap_gen.py (either in libraries or default_instrumentations)
    """
    if not package_name:
        return False

    # Check default instrumentations
    for default_instr in gen_default_instrumentations:
        if isinstance(default_instr, str):
            default_pkg_name = extract_package_name_from_requirement(
                default_instr
            )
            if package_names_match(default_pkg_name, package_name):
                return True

    # Check libraries mapping
    for lib_mapping in gen_libraries:
        instrumentation = lib_mapping.get("instrumentation", "")
        if isinstance(instrumentation, str):
            instr_pkg_name = extract_package_name_from_requirement(
                instrumentation
            )
            if package_names_match(instr_pkg_name, package_name):
                return True

    return False


def _is_loongsuite_package(package_name: str) -> bool:
    """Check if package is a loongsuite package (installed from GitHub Release tar)"""
    return package_name.startswith("loongsuite-")


def _get_desired_instrumentation_requirements(
    blacklist: Optional[Set[str]] = None,
    whitelist: Optional[Set[str]] = None,
    auto_detect: bool = False,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Get desired instrumentation packages from bootstrap_gen with filtering.

    Returns:
        (tar_packages, pypi_packages)
        - tar_packages: loongsuite-* packages to install from GitHub Release tar.gz
        - pypi_packages: opentelemetry-* packages to install from PyPI
    """
    blacklist = blacklist or set()
    whitelist = whitelist or set()
    tar_packages: List[Tuple[str, str]] = []
    pypi_packages: List[Tuple[str, str]] = []

    def _should_include(
        pkg_name: str, target_libraries: List[str], is_default: bool
    ) -> bool:
        if blacklist and pkg_name in blacklist:
            return False
        if whitelist and pkg_name not in whitelist:
            return False
        if is_default:
            return True
        if auto_detect and target_libraries:
            return any(_is_library_installed(lib) for lib in target_libraries)
        return not auto_detect

    seen: Set[str] = set()
    for default_instr in gen_default_instrumentations:
        if isinstance(default_instr, str):
            pkg_name = extract_package_name_from_requirement(default_instr)
            if pkg_name not in seen and _should_include(pkg_name, [], True):
                seen.add(pkg_name)
                if _is_loongsuite_package(pkg_name):
                    tar_packages.append((pkg_name, default_instr))
                else:
                    pypi_packages.append((pkg_name, default_instr))

    for lib_mapping in gen_libraries:
        instrumentation = lib_mapping.get("instrumentation", "")
        if isinstance(instrumentation, str):
            pkg_name = extract_package_name_from_requirement(instrumentation)
            target_lib = lib_mapping.get("library", "")
            target_libraries = [target_lib] if target_lib else []
            if pkg_name not in seen and _should_include(
                pkg_name, target_libraries, False
            ):
                seen.add(pkg_name)
                if _is_loongsuite_package(pkg_name):
                    tar_packages.append((pkg_name, instrumentation))
                else:
                    pypi_packages.append((pkg_name, instrumentation))

    return tar_packages, pypi_packages


def get_target_libraries_from_bootstrap_gen(
    package_name: str,
) -> Tuple[List[str], bool]:
    """
    Get target library requirements from bootstrap_gen.py.

    This function uses the pre-generated bootstrap_gen.py file to get
    target library information, similar to opentelemetry-bootstrap.

    Args:
        package_name: Name of the instrumentation package (e.g., "opentelemetry-instrumentation-redis")
                      May contain hyphens or underscores, will be normalized

    Returns:
        Tuple of (target_libraries list, is_default_instrumentation bool)
        target_libraries contains library requirement strings (e.g., ["redis >= 2.6"])
        is_default_instrumentation is True if this is a default instrumentation
    """
    if not package_name:
        return [], False

    # Check if it's a default instrumentation
    for default_instr in gen_default_instrumentations:
        if isinstance(default_instr, str):
            default_pkg_name = extract_package_name_from_requirement(
                default_instr
            )
            if package_names_match(default_pkg_name, package_name):
                return [], True

    # Look up in libraries mapping
    target_libraries = []
    for lib_mapping in gen_libraries:
        instrumentation = lib_mapping.get("instrumentation", "")
        if isinstance(instrumentation, str):
            instr_pkg_name = extract_package_name_from_requirement(
                instrumentation
            )
            if package_names_match(instr_pkg_name, package_name):
                target_lib = lib_mapping.get("library", "")
                if target_lib and isinstance(target_lib, str):
                    target_libraries.append(target_lib)

    return target_libraries, False


def check_dependency_compatibility(
    whl_path: Path, skip_version_check: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Check if package dependencies are compatible with installed packages

    Args:
        whl_path: Path to whl file
        skip_version_check: If True, skip version compatibility check

    Returns:
        (is_compatible, conflict_message)
        is_compatible: True if compatible, False otherwise
        conflict_message: Description of conflict if incompatible, None otherwise
    """
    if skip_version_check:
        return True, None

    metadata = get_metadata_from_whl(whl_path)
    if not metadata or "requires_dist" not in metadata:
        return True, None

    # Key packages to check compatibility
    key_packages = {
        "opentelemetry-instrumentation",
        "opentelemetry-semantic-conventions",
    }

    conflicts = []
    for req_str in metadata.get("requires_dist", []):
        try:
            req = Requirement(req_str)
            if req.name.lower() in key_packages:
                installed_version = get_installed_package_version(req.name)
                if installed_version:
                    # Check if installed version satisfies requirement
                    if not req.specifier.contains(installed_version):
                        conflicts.append(
                            f"{req.name} {installed_version} does not satisfy {req_str}"
                        )
        except Exception:
            # If parsing fails, assume compatible to avoid false positives
            continue

    if conflicts:
        conflict_msg = "; ".join(conflicts)
        return False, conflict_msg

    return True, None


def check_python_version_compatibility(
    whl_path: Path, current_version: Tuple[int, int]
) -> Tuple[bool, Optional[str]]:
    """
    Check if current Python version is compatible with whl file requirements

    Args:
        whl_path: Path to whl file
        current_version: Current Python version as (major, minor) tuple

    Returns:
        (is_compatible, requirement_string)
        is_compatible: True if compatible, False otherwise
        requirement_string: Python requirement string if found, None otherwise
    """
    requirement_str = get_python_requirement_from_whl(whl_path)

    if not requirement_str:
        # If no requirement found, assume compatible
        return True, None

    try:
        # Parse the requirement string
        spec = SpecifierSet(requirement_str)
        # Convert current version to string format
        current_version_str = f"{current_version[0]}.{current_version[1]}"
        # Check if current version satisfies the requirement
        is_compatible = spec.contains(current_version_str)
        return is_compatible, requirement_str
    except Exception:
        # If parsing fails, assume compatible to avoid false positives
        return True, requirement_str


def download_file(url: str, dest: Path) -> Path:
    """Download file to specified path"""
    logger.info(f"Downloading file: {url}")
    urllib.request.urlretrieve(url, dest)
    logger.info(f"Download completed: {dest}")
    return dest


def extract_tar(tar_path: Path, extract_dir: Path) -> List[Path]:
    """Extract tar.gz file, return all whl file paths"""
    logger.info(f"Extracting tar file: {tar_path} -> {extract_dir}")

    whl_files = []
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)

        # Find all whl files
        for member in tar.getmembers():
            if member.name.endswith(".whl"):
                whl_path = extract_dir / member.name
                if whl_path.exists():
                    whl_files.append(whl_path)

    logger.info(f"Extraction completed, found {len(whl_files)} whl files")
    return sorted(whl_files)


def filter_packages(
    whl_files: List[Path],
    blacklist: Optional[Set[str]] = None,
    whitelist: Optional[Set[str]] = None,
    skip_version_check: bool = False,
    auto_detect: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """
    Filter packages based on blacklist/whitelist, Python version compatibility,
    dependency version compatibility, and optionally auto-detect installed libraries

    Args:
        whl_files: List of whl file paths
        blacklist: blacklist (do not install these packages)
        whitelist: whitelist (only install these packages if specified)
        skip_version_check: If True, skip dependency version compatibility check
        auto_detect: If True, only install instrumentation packages if their target libraries are installed

    Returns:
        (base dependency packages list, instrumentation packages list)
    """
    base_packages = []
    instrumentation_packages = []

    blacklist = blacklist or set()
    whitelist = whitelist or set()

    # Get current Python version
    current_version = (sys.version_info.major, sys.version_info.minor)
    current_version_str = f"{current_version[0]}.{current_version[1]}"

    logger.info(f"Scanning {len(whl_files)} packages for installation...")
    if auto_detect:
        logger.info(
            "Auto-detect mode enabled: will only install instrumentations for detected libraries"
        )

    for whl_file in whl_files:
        package_name = get_package_name_from_whl(whl_file)

        # Check blacklist
        if blacklist and package_name in blacklist:
            logger.info(f"Skipping {package_name} (blacklist)")
            continue

        # Check whitelist
        if whitelist and package_name not in whitelist:
            logger.info(f"Skipping {package_name} (not in whitelist)")
            continue

        # Check Python version compatibility (only for instrumentations in bootstrap_gen.py)
        # Base dependencies and utility packages are installed without Python version check
        is_instrumentation = _is_instrumentation_in_bootstrap_gen(package_name)
        if is_instrumentation:
            is_compatible, requirement_str = (
                check_python_version_compatibility(whl_file, current_version)
            )
            if not is_compatible:
                logger.info(
                    f"Skipping {package_name} (Python version incompatible: requires {requirement_str}, current: {current_version_str})"
                )
                continue

        # Check dependency version compatibility (only for base dependencies)
        # Instrumentation packages will be checked by pip during installation
        if package_name in BASE_DEPENDENCIES_PYPI:
            is_dep_compatible, conflict_msg = check_dependency_compatibility(
                whl_file, skip_version_check
            )
            if not is_dep_compatible:
                logger.warning(
                    f"Skipping {package_name} (dependency version incompatible: {conflict_msg})"
                )
                continue

        # Classify: base dependencies vs instrumentation
        if package_name in BASE_DEPENDENCIES_PYPI:
            base_packages.append(whl_file)
        else:
            # For instrumentation packages, check if auto-detect is enabled
            if auto_detect:
                target_libraries, is_default = (
                    get_target_libraries_from_bootstrap_gen(package_name)
                )

                # Default instrumentations are always installed (like opentelemetry-bootstrap)
                if is_default:
                    logger.info(
                        f"Will install {package_name} (default instrumentation)"
                    )
                    instrumentation_packages.append(whl_file)
                elif target_libraries:
                    # Check if any target library is installed
                    library_installed = False
                    installed_libs = []
                    not_installed_libs = []
                    for lib_req in target_libraries:
                        if _is_library_installed(lib_req):
                            library_installed = True
                            try:
                                req = Requirement(lib_req)
                                installed_libs.append(req.name)
                            except Exception:
                                installed_libs.append(lib_req)
                        else:
                            try:
                                req = Requirement(lib_req)
                                not_installed_libs.append(req.name)
                            except Exception:
                                not_installed_libs.append(lib_req)

                    if library_installed:
                        logger.info(
                            f"Will install {package_name} (detected libraries: {', '.join(installed_libs)})"
                        )
                        instrumentation_packages.append(whl_file)
                    else:
                        logger.info(
                            f"Skipping {package_name} (required libraries not installed: {', '.join(not_installed_libs)})"
                        )
                        continue
                else:
                    # No mapping found in bootstrap_gen.py, skip it
                    logger.info(
                        f"Skipping {package_name} (no target libraries mapping in bootstrap_gen.py)"
                    )
                    continue
            else:
                # Auto-detect disabled, install all instrumentation packages
                logger.info(f"Will install {package_name}")
                instrumentation_packages.append(whl_file)

    return base_packages, instrumentation_packages


def install_packages(
    whl_files: List[Path],
    find_links_dir: Path,
    upgrade: bool = False,
    extra_requirements: Optional[List[str]] = None,
):
    """Install packages using pip. extra_requirements are installed from PyPI."""
    if not whl_files and not extra_requirements:
        logger.warning("No packages to install")
        return

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--find-links",
        str(find_links_dir),
    ]

    if upgrade:
        cmd.append("--upgrade")

    # Add whl files (from tar) and extra requirements (from PyPI)
    cmd.extend([str(whl) for whl in whl_files])
    if extra_requirements:
        cmd.extend(extra_requirements)

    logger.info(f"Executing install command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info("Installation completed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Installation failed: {e}")
        raise


def get_installed_loongsuite_packages() -> List[str]:
    """
    Get list of installed loongsuite and opentelemetry packages to uninstall

    Excludes:
    - loongsuite-distro
    - opentelemetry-api
    - opentelemetry-sdk
    - opentelemetry-instrumentation

    Returns:
        List of installed package names to uninstall
    """
    cmd = [sys.executable, "-m", "pip", "list", "--format=json"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        installed_packages = json_lib.loads(result.stdout)

        # Filter packages to uninstall
        packages_to_uninstall = []
        for pkg in installed_packages:
            name = pkg.get("name", "")
            name_lower = name.lower()

            # Skip excluded packages
            if name_lower in UNINSTALL_EXCLUDED_PACKAGES:
                continue

            # Include loongsuite-* packages (except loongsuite-distro)
            if name_lower.startswith("loongsuite-"):
                packages_to_uninstall.append(name)
            # Include opentelemetry-* packages (except opentelemetry-api and opentelemetry-sdk)
            elif name_lower.startswith("opentelemetry-"):
                packages_to_uninstall.append(name)

        return packages_to_uninstall
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get installed packages: {e}")
        raise
    except json_lib.JSONDecodeError as e:
        logger.error(f"Failed to parse pip list output: {e}")
        raise


def uninstall_packages(package_names: List[str], yes: bool = False):
    """Uninstall packages using pip"""
    if not package_names:
        logger.warning("No packages to uninstall")
        return

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
    ]

    if yes:
        cmd.append("-y")

    # Add all package names
    cmd.extend(package_names)

    logger.info(f"Executing uninstall command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info("Uninstallation completed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Uninstallation failed: {e}")
        raise


def resolve_tar_path(
    tar_path: Union[Path, str],
) -> Tuple[Path, Optional[Path]]:
    """
    Resolve tar path, downloading from URI if necessary

    Args:
        tar_path: tar file path or URI (can be Path or str)

    Returns:
        (local_tar_path, temp_dir_to_cleanup)
        local_tar_path: Path to local tar file
        temp_dir_to_cleanup: Path to temporary directory to clean up (None if not downloaded)
    """
    tar_path_str = str(tar_path)
    if tar_path_str.startswith(("http://", "https://")):
        # Download from URI
        temp_dir = Path(tempfile.mkdtemp(prefix="loongsuite-download-"))
        temp_tar = temp_dir / "loongsuite.tar.gz"
        download_file(tar_path_str, temp_tar)
        return temp_tar, temp_dir
    else:
        tar_path = Path(tar_path)
        if not tar_path.exists():
            raise FileNotFoundError(f"Tar file does not exist: {tar_path}")
        return tar_path, None


def get_package_names_from_tar(
    tar_path: Path,
    blacklist: Optional[Set[str]] = None,
    whitelist: Optional[Set[str]] = None,
) -> List[str]:
    """
    Extract package names from tar file

    Args:
        tar_path: Path to tar file
        blacklist: blacklist (do not include these packages)
        whitelist: whitelist (only include these packages if specified)

    Returns:
        List of package names
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="loongsuite-"))
    try:
        whl_files = extract_tar(tar_path, temp_dir)
        if not whl_files:
            raise ValueError("No whl files found in tar file")

        base_packages, instrumentation_packages = filter_packages(
            whl_files, blacklist, whitelist, auto_detect=False
        )

        # Get package names
        package_names = []
        for whl in base_packages + instrumentation_packages:
            package_name = get_package_name_from_whl(whl)
            package_names.append(package_name)

        return package_names
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def install_from_tar(
    tar_path: Union[Path, str],
    blacklist: Optional[Set[str]] = None,
    whitelist: Optional[Set[str]] = None,
    upgrade: bool = False,
    keep_temp: bool = False,
    skip_version_check: bool = False,
    auto_detect: bool = False,
):
    """
    Two-phase installation from tar package:
    1. Install loongsuite-* packages from GitHub Release tar.gz (GenAI instrumentations)
    2. Install opentelemetry-* packages from PyPI (standard instrumentations)

    Args:
        tar_path: tar file path or URI (can be Path or str)
        blacklist: blacklist (do not install these packages)
        whitelist: whitelist (only install these packages if specified)
        upgrade: whether to upgrade already installed packages
        keep_temp: whether to keep temporary directory
        skip_version_check: If True, skip dependency version compatibility check
        auto_detect: If True, only install instrumentation packages if their target libraries are installed
    """
    # Resolve tar path (download from URI if necessary)
    local_tar_path, temp_tar_dir = resolve_tar_path(tar_path)

    # Create temporary directory for extraction
    temp_dir = Path(tempfile.mkdtemp(prefix="loongsuite-"))

    try:
        logger.info("Extracting packages from tar file...")
        # Extract tar file
        whl_files = extract_tar(local_tar_path, temp_dir)

        if not whl_files:
            raise ValueError("No whl files found in tar file")

        logger.info(f"Found {len(whl_files)} packages in tar file")

        # Filter packages from tar (loongsuite-* packages)
        logger.info("Filtering packages...")
        base_packages, instrumentation_packages = filter_packages(
            whl_files, blacklist, whitelist, skip_version_check, auto_detect
        )

        # Get desired packages from bootstrap_gen
        tar_desired, pypi_desired = _get_desired_instrumentation_requirements(
            blacklist, whitelist, auto_detect
        )

        # Build package name set from tar
        tar_package_names = {
            normalize_package_name(get_package_name_from_whl(w))
            for w in whl_files
        }

        # loongsuite-* packages from tar (already filtered)
        tar_packages = base_packages + instrumentation_packages

        # opentelemetry-* packages from PyPI (use requirement string with version)
        pypi_requirements: List[str] = []
        for pkg_name, req_str in pypi_desired:
            norm_name = normalize_package_name(pkg_name)
            if norm_name not in tar_package_names:
                # Use full requirement string (e.g., "opentelemetry-instrumentation-flask==0.60b1")
                pypi_requirements.append(req_str)

        if not tar_packages and not pypi_requirements:
            logger.warning("No packages to install after filtering")
            return

        # Phase 1: Install from tar.gz (loongsuite-* packages)
        logger.info("=" * 50)
        logger.info("Phase 1: Installing loongsuite-* packages from tar...")
        logger.info("=" * 50)
        if tar_packages:
            logger.info(f"Will install {len(tar_packages)} packages from tar:")
            for pkg in tar_packages:
                pkg_name = get_package_name_from_whl(pkg)
                logger.info(f"  - {pkg_name}")
            install_packages(tar_packages, temp_dir, upgrade)
        else:
            logger.info("No loongsuite-* packages to install from tar")

        # Phase 2: Install from PyPI (opentelemetry-* packages)
        logger.info("=" * 50)
        logger.info(
            "Phase 2: Installing opentelemetry-* packages from PyPI..."
        )
        logger.info("=" * 50)
        if pypi_requirements:
            logger.info(
                f"Will install {len(pypi_requirements)} packages from PyPI:"
            )
            for req in pypi_requirements:
                logger.info(f"  - {req}")
            install_packages(
                [], temp_dir, upgrade, extra_requirements=pypi_requirements
            )
        else:
            logger.info("No opentelemetry-* packages to install from PyPI")

        logger.info("=" * 50)
        logger.info("Installation completed successfully!")
        logger.info("=" * 50)

    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if temp_tar_dir and temp_tar_dir.exists():
                shutil.rmtree(temp_tar_dir, ignore_errors=True)
        else:
            logger.info(f"Temporary directory kept at: {temp_dir}")
            if temp_tar_dir:
                logger.info(f"Downloaded tar file kept at: {local_tar_path}")


def uninstall_loongsuite_packages(
    blacklist: Optional[Set[str]] = None,
    whitelist: Optional[Set[str]] = None,
    yes: bool = False,
):
    """
    Uninstall installed loongsuite packages

    Args:
        blacklist: blacklist (do not uninstall these packages)
        whitelist: whitelist (only uninstall these packages if specified)
        yes: automatically confirm uninstallation
    """
    # Get installed loongsuite packages
    installed_packages = get_installed_loongsuite_packages()

    if not installed_packages:
        logger.warning("No loongsuite packages found installed")
        return

    # Apply blacklist/whitelist filters
    blacklist = blacklist or set()
    whitelist = whitelist or set()

    package_names = []
    for pkg in installed_packages:
        # Check blacklist
        if blacklist and pkg in blacklist:
            logger.debug(f"Skipping package (blacklist): {pkg}")
            continue

        # Check whitelist
        if whitelist and pkg not in whitelist:
            logger.debug(f"Skipping package (not in whitelist): {pkg}")
            continue

        package_names.append(pkg)

    if not package_names:
        logger.warning("No packages to uninstall after filtering")
        return

    logger.info(f"Will uninstall {len(package_names)} packages:")
    for name in package_names:
        logger.info(f"  - {name}")

    # Uninstall
    uninstall_packages(package_names, yes)


def get_latest_release_url(
    repo: str = "alibaba/loongsuite-python",
) -> str:
    """Get latest release tar.gz URL from GitHub API"""
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    logger.info(f"Fetching latest release: {api_url}")

    try:
        with urllib.request.urlopen(api_url) as response:
            data = json_lib.loads(response.read())
            for asset in data.get("assets", []):
                if asset["name"].endswith(".tar.gz"):
                    return asset["browser_download_url"]

        # If no asset found, try to build URL from tag
        tag = data.get("tag_name", "").lstrip("v")
        return f"https://github.com/{repo}/releases/download/{data.get('tag_name')}/loongsuite-python-{tag}.tar.gz"
    except Exception as e:
        logger.error(f"Failed to fetch latest release: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="""
        LoongSuite Bootstrap - Install/Uninstall loongsuite Python Agent from tar package

        This tool installs or uninstalls all loongsuite components from tar.gz file.
        Supports blacklist/whitelist to control which instrumentations to install/uninstall.
        """
    )

    parser.add_argument(
        "-a",
        "--action",
        choices=["install", "uninstall"],
        required=True,
        help="action type: install to install packages, uninstall to uninstall packages",
    )

    # Common arguments
    parser.add_argument(
        "--blacklist",
        type=Path,
        help="blacklist file path (one package name per line, do not install/uninstall these packages)",
    )
    parser.add_argument(
        "--whitelist",
        type=Path,
        help="whitelist file path (one package name per line, only install/uninstall these packages)",
    )

    # Install-specific arguments
    install_group = parser.add_argument_group("install options")
    install_group.add_argument(
        "-t",
        "--tar",
        type=str,
        help="tar package path or URI (required for install action, supports http:// and https://)",
    )
    install_group.add_argument(
        "-v",
        "--version",
        type=str,
        help="version number, download from GitHub Releases (e.g., 1.0.0) (for install action)",
    )
    install_group.add_argument(
        "--latest",
        action="store_true",
        help="install latest version (from GitHub Releases) (for install action)",
    )
    install_group.add_argument(
        "--upgrade",
        action="store_true",
        help="upgrade already installed packages (for install action)",
    )
    install_group.add_argument(
        "--keep-temp",
        action="store_true",
        help="keep temporary directory (for debugging)",
    )
    install_group.add_argument(
        "--force",
        action="store_true",
        help="force installation even if dependency versions are incompatible",
    )
    install_group.add_argument(
        "--auto-detect",
        action="store_true",
        help="only install instrumentation packages if their target libraries are installed (similar to opentelemetry-bootstrap)",
    )
    install_group.add_argument(
        "--verbose",
        action="store_true",
        help="enable verbose debug logging",
    )

    # Uninstall-specific arguments
    uninstall_group = parser.add_argument_group("uninstall options")
    uninstall_group.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="automatically confirm uninstallation (for uninstall action)",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose or (
        hasattr(args, "action")
        and args.action == "install"
        and hasattr(args, "auto_detect")
        and args.auto_detect
    ):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s: %(message)s",
            force=True,
        )
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s: %(message)s", force=True
        )
        logger.setLevel(logging.INFO)

    # Load blacklist/whitelist
    blacklist = load_list_file(args.blacklist) if args.blacklist else None
    whitelist = load_list_file(args.whitelist) if args.whitelist else None

    if blacklist:
        logger.info(f"Blacklist: {len(blacklist)} packages")
    if whitelist:
        logger.info(f"Whitelist: {len(whitelist)} packages")

    if args.action == "install":
        # Determine tar file path
        tar_path = None
        if args.tar:
            tar_path = args.tar
        elif args.version:
            tar_path = f"https://github.com/alibaba/loongsuite-python/releases/download/v{args.version}/loongsuite-python-{args.version}.tar.gz"
        elif args.latest:
            tar_path = get_latest_release_url()
        else:
            parser.error(
                "For install action, must specify one of --tar, --version, or --latest"
            )

        # Install
        install_from_tar(
            tar_path,
            blacklist=blacklist,
            whitelist=whitelist,
            upgrade=args.upgrade,
            keep_temp=args.keep_temp,
            skip_version_check=args.force,
            auto_detect=args.auto_detect,
        )

    elif args.action == "uninstall":
        # Uninstall installed loongsuite packages
        uninstall_loongsuite_packages(
            blacklist=blacklist,
            whitelist=whitelist,
            yes=args.yes,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    main()
