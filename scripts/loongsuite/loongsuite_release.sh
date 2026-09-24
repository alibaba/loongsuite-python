#!/usr/bin/env bash

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

#
# LoongSuite Release Script
#
# Unified script for both local and CI release workflows.
#
# Modes:
#   Default (no --dry-run):
#     1. Create release/{version} branch from main
#     2. Generate bootstrap_gen.py
#     3. Rename package names in pyproject.toml (opentelemetry-util-genai -> loongsuite-otel-util-genai)
#     3.6 Pin __version__ in managed packages to the release version (PyPI train)
#     4. Build PyPI + GitHub Release packages
#     5. Verify artifacts
#     6. Collect changelogs into release notes
#     7. Archive changelogs (Unreleased -> versioned)
#     8. Commit & push release branch
#     9. (Optional) Installation verification
#     10. (Optional) Create GitHub Release via gh CLI
#     11. Create post-release PR to main (archive changelogs + bump dev versions)
#
#   --dry-run:
#     Runs steps 2, 4-6, 9 only (no branch creation, no rename, no changelog archive, no commit, no release).
#
# Usage:
#   # Local dry run (validate build)
#   ./scripts/loongsuite/loongsuite_release.sh -l 0.1.0 -u 0.60b1 --dry-run
#
#   # Local full release
#   ./scripts/loongsuite/loongsuite_release.sh -l 0.1.0 -u 0.60b1
#
#   # CI release (skip GitHub Release creation - done in separate job)
#   ./scripts/loongsuite/loongsuite_release.sh -l 0.1.0 -u 0.60b1 --skip-github-release
#
set -e

LOONGSUITE_VERSION=""
UPSTREAM_VERSION=""
DRY_RUN=false
SKIP_INSTALL=false
SKIP_PYPI=false
SKIP_GITHUB_RELEASE=false
SKIP_POST_RELEASE_PR=false
GIT_REMOTE="origin"

while [[ $# -gt 0 ]]; do
  case $1 in
    -l|--loongsuite-version)
      LOONGSUITE_VERSION="$2"
      shift 2
      ;;
    -u|--upstream-version)
      UPSTREAM_VERSION="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=true
      shift
      ;;
    --skip-pypi)
      SKIP_PYPI=true
      shift
      ;;
    --skip-github-release)
      SKIP_GITHUB_RELEASE=true
      shift
      ;;
    --skip-post-release-pr)
      SKIP_POST_RELEASE_PR=true
      shift
      ;;
    --remote)
      GIT_REMOTE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --loongsuite-version <ver> --upstream-version <ver> [options]"
      echo ""
      echo "Required:"
      echo "  -l, --loongsuite-version  Version for loongsuite-* packages (e.g., 0.1.0)"
      echo "  -u, --upstream-version    Version for opentelemetry-* packages (e.g., 0.60b1)"
      echo ""
      echo "Options:"
      echo "  --dry-run             Validate only, no branch/commit/release"
      echo "  --skip-install        Skip installation verification"
      echo "  --skip-pypi           Skip PyPI package build"
      echo "  --skip-github-release Skip creating GitHub Release (for CI)"
      echo "  --skip-post-release-pr Skip creating post-release PR to main"
      echo "  --remote <name>       Git remote name (default: origin)"
      echo "  -h, --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$LOONGSUITE_VERSION" ]]; then
  echo "ERROR: --loongsuite-version is required"
  exit 1
fi
if [[ -z "$UPSTREAM_VERSION" ]]; then
  echo "ERROR: --upstream-version is required"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RELEASE_BRANCH="release/${LOONGSUITE_VERSION}"
TAR_NAME="loongsuite-python-${LOONGSUITE_VERSION}.tar.gz"
TAR_PATH="${REPO_ROOT}/dist/${TAR_NAME}"
PYPI_DIST_DIR="${REPO_ROOT}/dist-pypi"
RELEASE_NOTES_FILE="${REPO_ROOT}/dist/release-notes.md"
DRYRUN_VENV="${REPO_ROOT}/.venv_loongsuite_dryrun"
TODAY=$(date -u +%Y-%m-%d)

MODE="RELEASE"
if [[ "$DRY_RUN" == "true" ]]; then
  MODE="DRY RUN"
fi

echo "=========================================="
echo "LoongSuite Release ($MODE)"
echo "=========================================="
echo "LoongSuite version: $LOONGSUITE_VERSION"
echo "Upstream version:   $UPSTREAM_VERSION"
echo "Repo root:          $REPO_ROOT"
if [[ "$DRY_RUN" != "true" ]]; then
  echo "Release branch:     $RELEASE_BRANCH"
fi
echo ""

# ── Step 1: Install build dependencies ──────────────────────────────────────
echo ">>> Step 1: Installing build dependencies..."
python -m pip install -q -r pkg-requirements.txt 2>/dev/null || {
  echo "    Installing dependencies from pkg-requirements.txt..."
  python -m pip install -r pkg-requirements.txt
}
echo "    OK"
echo ""

# ── Step 2: Create release branch (skip in dry-run) ────────────────────────
if [[ "$DRY_RUN" != "true" ]]; then
  echo ">>> Step 2: Creating release branch ${RELEASE_BRANCH}..."
  git checkout -b "$RELEASE_BRANCH" main
  echo "    OK: Branch $RELEASE_BRANCH created from main"
  echo ""
else
  echo ">>> Step 2: Skipped (dry-run mode)"
  echo ""
fi

# ── Step 3: Generate bootstrap_gen.py ───────────────────────────────────────
echo ">>> Step 3: Generating bootstrap_gen.py..."
python scripts/loongsuite/generate_loongsuite_bootstrap.py \
  --upstream-version "$UPSTREAM_VERSION" \
  --loongsuite-version "$LOONGSUITE_VERSION"

echo "    OK: Generated bootstrap_gen.py"
echo "    Preview (first 20 lines):"
head -20 loongsuite-distro/src/loongsuite/distro/bootstrap_gen.py | sed 's/^/    /'
echo ""

# ── Step 3.5: Rename packages in pyproject.toml (skip in dry-run) ──────────
if [[ "$DRY_RUN" != "true" ]]; then
  echo ">>> Step 3.5: Renaming opentelemetry-util-genai -> loongsuite-otel-util-genai in pyproject.toml..."
  python scripts/loongsuite/collect_loongsuite_changelog.py \
    --version "$LOONGSUITE_VERSION"
  echo "    OK"
  echo ""

  echo ">>> Step 3.6: Pinning package __version__ to ${LOONGSUITE_VERSION} (release train)..."
  python scripts/loongsuite/collect_loongsuite_changelog.py \
    --pin-release-versions \
    --version "$LOONGSUITE_VERSION"
  echo "    OK"
  echo ""
else
  echo ">>> Step 3.5: Skipped (dry-run mode)"
  echo ""
  echo ">>> Step 3.6: Skipped (dry-run mode)"
  echo ""
fi

# ── Step 4: Build PyPI packages ────────────────────────────────────────────
rm -rf "$PYPI_DIST_DIR"
mkdir -p "$PYPI_DIST_DIR"

if [[ "$SKIP_PYPI" != "true" ]]; then
  echo ">>> Step 4: Building PyPI packages..."
  python scripts/loongsuite/build_loongsuite_package.py \
    --build-pypi \
    --version "$LOONGSUITE_VERSION"

  cp dist/*.whl "$PYPI_DIST_DIR/" 2>/dev/null || true

  echo "    OK: PyPI packages built"
  echo "    Packages:"
  ls "$PYPI_DIST_DIR"/*.whl 2>/dev/null | while read f; do echo "      - $(basename "$f")"; done
  echo ""
else
  echo ">>> Step 4: Skipped (--skip-pypi)"
  echo ""
fi

# ── Step 5: Build GitHub Release packages ──────────────────────────────────
echo ">>> Step 5: Building GitHub Release packages..."
python scripts/loongsuite/build_loongsuite_package.py \
  --build-github-release \
  --version "$LOONGSUITE_VERSION"

if [[ ! -f "$TAR_PATH" ]]; then
  echo "    ERROR: Build failed, $TAR_PATH not found"
  exit 1
fi
echo "    OK: $TAR_PATH ($(du -h "$TAR_PATH" | cut -f1))"
echo ""

# ── Step 6: Verify tar contents ────────────────────────────────────────────
echo ">>> Step 6: Verifying tar contents..."

if tar -tzf "$TAR_PATH" | grep -q "loongsuite_otel_util_genai"; then
  echo "    WARN: loongsuite-otel-util-genai in tar (should be on PyPI only)"
else
  echo "    OK: loongsuite-otel-util-genai not in tar (correct, on PyPI)"
fi

if tar -tzf "$TAR_PATH" | grep -q "opentelemetry_util_genai"; then
  echo "    ERROR: opentelemetry-util-genai should NOT be in tar"
  exit 1
else
  echo "    OK: opentelemetry-util-genai not in tar"
fi

if tar -tzf "$TAR_PATH" | grep -q "loongsuite_instrumentation"; then
  echo "    OK: loongsuite-instrumentation-* packages in tar"
else
  echo "    WARN: No loongsuite-instrumentation-* packages found in tar"
fi

if tar -tzf "$TAR_PATH" | grep -q "opentelemetry_instrumentation_flask"; then
  echo "    WARN: opentelemetry-instrumentation-flask in tar (should be from PyPI)"
else
  echo "    OK: opentelemetry-instrumentation-flask not in tar (from PyPI)"
fi

echo "    Package count: $(tar -tzf "$TAR_PATH" | wc -l | tr -d ' ')"
echo "    Contents:"
tar -tzf "$TAR_PATH" | head -20 | sed 's/^/      /'
echo ""

# ── Step 7: Collect changelog & generate release notes ─────────────────────
echo ">>> Step 7: Collecting changelogs and generating release notes..."
python scripts/loongsuite/collect_loongsuite_changelog.py \
  --collect \
  --version "$LOONGSUITE_VERSION" \
  --upstream-version "$UPSTREAM_VERSION" \
  --output "$RELEASE_NOTES_FILE"

echo "    OK: $RELEASE_NOTES_FILE"
echo "    Preview:"
head -30 "$RELEASE_NOTES_FILE" | sed 's/^/    /'
echo ""

# ── Step 8: Archive changelogs (skip in dry-run) ──────────────────────────
if [[ "$DRY_RUN" != "true" ]]; then
  echo ">>> Step 8: Archiving changelogs (Unreleased -> Version $LOONGSUITE_VERSION)..."
  python scripts/loongsuite/collect_loongsuite_changelog.py \
    --archive \
    --version "$LOONGSUITE_VERSION" \
    --date "$TODAY"
  echo "    OK"
  echo ""
else
  echo ">>> Step 8: Skipped (dry-run mode)"
  echo ""
fi

# ── Step 9: Commit & push release branch (skip in dry-run) ────────────────
if [[ "$DRY_RUN" != "true" ]]; then
  echo ">>> Step 9: Running precommit checks..."
  tox -e precommit || echo "    WARN: precommit had issues, please review"
  echo ""

  echo ">>> Step 9: Committing changes to ${RELEASE_BRANCH}..."
  git add -A
  git commit -m "release: LoongSuite v${LOONGSUITE_VERSION}

- Generate bootstrap_gen.py with upstream=${UPSTREAM_VERSION}, loongsuite=${LOONGSUITE_VERSION}
- Pin managed __version__ files to ${LOONGSUITE_VERSION} for PyPI packages
- Archive changelogs for version ${LOONGSUITE_VERSION}"

  echo "    Pushing to ${GIT_REMOTE}/${RELEASE_BRANCH}..."
  git push "$GIT_REMOTE" "$RELEASE_BRANCH"
  echo "    OK"
  echo ""
else
  echo ">>> Step 9: Skipped (dry-run mode)"
  echo ""
fi

# ── Step 10: Install verification (optional) ──────────────────────────────
if [[ "$SKIP_INSTALL" == "true" ]]; then
  echo ">>> Step 10: Skipped (--skip-install)"
elif [[ "$SKIP_PYPI" == "true" ]]; then
  echo ">>> Step 10: Skipped (requires PyPI build, which was skipped)"
else
  echo ">>> Step 10: Install verification (temp venv)..."
  rm -rf "$DRYRUN_VENV"
  python -m venv "$DRYRUN_VENV"
  source "$DRYRUN_VENV/bin/activate"

  echo "    Installing loongsuite-distro from local..."
  pip install -q -e ./loongsuite-distro

  UTIL_WHL=$(ls "$PYPI_DIST_DIR"/loongsuite_otel_util_genai-*.whl 2>/dev/null | head -1)
  if [[ -n "$UTIL_WHL" ]]; then
    echo "    Pre-installing loongsuite-otel-util-genai from local build..."
    pip install -q "$UTIL_WHL"
  else
    echo "    ERROR: loongsuite-otel-util-genai wheel not found in $PYPI_DIST_DIR"
    deactivate
    rm -rf "$DRYRUN_VENV"
    exit 1
  fi

  WHITELIST_FILE=$(mktemp)
  cat > "$WHITELIST_FILE" << 'WL'
loongsuite-instrumentation-dashscope
WL

  echo "    Running: loongsuite-bootstrap -a install --tar $TAR_PATH --whitelist $WHITELIST_FILE"
  if loongsuite-bootstrap -a install --tar "$TAR_PATH" --whitelist "$WHITELIST_FILE" 2>&1; then
    echo ""
    echo "    Verifying installed packages..."

    if pip show loongsuite-otel-util-genai &>/dev/null; then
      echo "    OK: loongsuite-otel-util-genai installed ($(pip show loongsuite-otel-util-genai | grep Version:))"
    else
      echo "    WARN: loongsuite-otel-util-genai not installed"
    fi

    if pip show opentelemetry-util-genai &>/dev/null; then
      echo "    WARN: opentelemetry-util-genai installed (may conflict)"
    else
      echo "    OK: opentelemetry-util-genai not installed (correct)"
    fi

    if pip show loongsuite-instrumentation-dashscope &>/dev/null; then
      echo "    OK: loongsuite-instrumentation-dashscope installed"
    else
      echo "    WARN: loongsuite-instrumentation-dashscope not installed"
    fi

    rm -f "$WHITELIST_FILE"
    deactivate
    rm -rf "$DRYRUN_VENV"
    echo "    OK: Install verification passed"
  else
    echo "    ERROR: loongsuite-bootstrap install failed"
    rm -f "$WHITELIST_FILE"
    deactivate
    rm -rf "$DRYRUN_VENV"
    exit 1
  fi
fi
echo ""

# ── Step 11: Create GitHub Release (optional) ─────────────────────────────
if [[ "$DRY_RUN" == "true" || "$SKIP_GITHUB_RELEASE" == "true" ]]; then
  echo ">>> Step 11: Skipped (dry-run or --skip-github-release)"
  echo ""
else
  echo ">>> Step 11: Creating GitHub Release..."
  if ! command -v gh &>/dev/null; then
    echo "    WARN: gh CLI not found, skipping GitHub Release creation."
    echo "    Run manually:"
    echo "      gh release create v$LOONGSUITE_VERSION \\"
    echo "        --title \"loongsuite-python $LOONGSUITE_VERSION\" \\"
    echo "        --notes-file $RELEASE_NOTES_FILE \\"
    echo "        $TAR_PATH"
  else
    gh release create "v${LOONGSUITE_VERSION}" \
      --title "loongsuite-python ${LOONGSUITE_VERSION}" \
      --notes-file "$RELEASE_NOTES_FILE" \
      "$TAR_PATH"
    echo "    OK: GitHub Release v${LOONGSUITE_VERSION} created"
  fi
  echo ""
fi

# ── Step 12: Create post-release PR to main ────────────────────────────
if [[ "$DRY_RUN" == "true" || "$SKIP_POST_RELEASE_PR" == "true" ]]; then
  echo ">>> Step 12: Skipped (dry-run or --skip-post-release-pr)"
  echo ""
else
  echo ">>> Step 12: Creating post-release PR to main..."
  POST_RELEASE_BRANCH="post-release/${LOONGSUITE_VERSION}"

  git checkout main
  git checkout -b "$POST_RELEASE_BRANCH"

  echo "    Archiving changelogs on main..."
  python scripts/loongsuite/collect_loongsuite_changelog.py \
    --archive \
    --version "$LOONGSUITE_VERSION" \
    --date "$TODAY"

  echo "    Bumping instrumentation-loongsuite versions to next dev..."
  python scripts/loongsuite/collect_loongsuite_changelog.py \
    --bump-dev \
    --version "$LOONGSUITE_VERSION"

  echo "    Running precommit checks..."
  tox -e precommit || echo "    WARN: precommit had issues, please review"

  git add -A
  git commit -m "chore: post-release v${LOONGSUITE_VERSION} - archive changelogs & bump dev versions

- Archive Unreleased changelogs as Version ${LOONGSUITE_VERSION}
- Bump instrumentation-loongsuite + loongsuite-distro + loongsuite-site-bootstrap versions to next dev"

  echo "    Pushing ${POST_RELEASE_BRANCH}..."
  git push "$GIT_REMOTE" "$POST_RELEASE_BRANCH"

  if command -v gh &>/dev/null; then
    gh pr create \
      --base main \
      --head "$POST_RELEASE_BRANCH" \
      --title "chore: post-release ${LOONGSUITE_VERSION} — archive changelogs & bump dev versions" \
      --body "## Post-release updates for loongsuite-python ${LOONGSUITE_VERSION}

Automated housekeeping after the \`${LOONGSUITE_VERSION}\` release:

- **Archive changelogs**: move \`Unreleased\` sections to \`Version ${LOONGSUITE_VERSION} (${TODAY})\` in all \`CHANGELOG\` files
- **Bump dev versions**: update \`instrumentation-loongsuite/**/version.py\`, \`loongsuite-distro/src/loongsuite/distro/version.py\`, and \`loongsuite-site-bootstrap/src/loongsuite_site_bootstrap/version.py\` to the next \`.dev\` iteration

> This PR was auto-generated by \`scripts/loongsuite/loongsuite_release.sh\`."
    echo "    OK: Post-release PR created"
  else
    echo "    WARN: gh CLI not found, skipping PR creation."
    echo "    Push completed. Create PR manually from ${POST_RELEASE_BRANCH} → main"
  fi

  # Switch back to the release branch
  git checkout "$RELEASE_BRANCH"
  echo ""
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo "=========================================="
echo "Complete ($MODE)"
echo "=========================================="
echo ""
echo "Artifacts:"
if [[ "$SKIP_PYPI" != "true" ]]; then
  echo "  PyPI packages (in $PYPI_DIST_DIR):"
  ls "$PYPI_DIST_DIR"/*.whl 2>/dev/null | while read f; do echo "    - $f"; done
fi
echo "  GitHub Release:"
echo "    - $TAR_PATH"
echo "  Release notes:"
echo "    - $RELEASE_NOTES_FILE"

if [[ "$DRY_RUN" == "true" ]]; then
  echo ""
  echo "This was a dry run. To perform the actual release, run without --dry-run."
  echo ""
  echo "  ./scripts/loongsuite/loongsuite_release.sh \\"
  echo "    -l $LOONGSUITE_VERSION -u $UPSTREAM_VERSION"
fi
echo ""
