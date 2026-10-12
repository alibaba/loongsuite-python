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
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPT_DIR = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_new_pypi_projects", SCRIPT_DIR / "select_new_pypi_projects.py"
)
select = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(SCRIPT_DIR), *sys.path]):
    SPEC.loader.exec_module(select)


@pytest.mark.parametrize("status", [404, 410])
def test_absent_project(status):
    error = urllib.error.HTTPError(
        "https://pypi.org/", status, "missing", {}, None
    )
    with (
        patch.object(select.urllib.request, "urlopen", side_effect=error),
        patch.object(select.time, "sleep") as sleep,
    ):
        assert select.project_exists_on_pypi("new-project", timeout=1) is False
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("https://pypi.org/", 429, "retry", {}, None),
        urllib.error.HTTPError("https://pypi.org/", 503, "retry", {}, None),
        OSError("network unavailable"),
    ],
)
def test_transient_lookup_retries_then_succeeds(error):
    response = MagicMock()
    response.__enter__.return_value.status = 200
    with (
        patch.object(
            select.urllib.request, "urlopen", side_effect=[error, response]
        ) as request,
        patch.object(select.time, "sleep") as sleep,
    ):
        assert select.project_exists_on_pypi("new-project", timeout=1) is True
    assert request.call_count == 2
    sleep.assert_called_once_with(1)


def test_lookup_retry_exhaustion_does_not_report_project_absent():
    with (
        patch.object(
            select.urllib.request, "urlopen", side_effect=OSError("offline")
        ) as request,
        patch.object(select.time, "sleep") as sleep,
    ):
        with pytest.raises(RuntimeError, match="PyPI project lookup failed"):
            select.project_exists_on_pypi("new-project", timeout=1)
    assert request.call_count == 3
    assert sleep.call_count == 2


@pytest.mark.parametrize("dry_run", [True, False])
def test_prune_and_github_outputs(tmp_path, monkeypatch, dry_run):
    existing = tmp_path / "existing_project-0.1-py3-none-any.whl"
    missing = tmp_path / "new_project-0.1-py3-none-any.whl"
    existing.touch()
    missing.touch()
    output = tmp_path / "outputs"
    output.write_text("previous=value\n")
    summary = tmp_path / "summary"
    args = [
        "select",
        "--dist-dir",
        str(tmp_path),
        "--github-output",
        str(output),
        "--summary",
        str(summary),
    ]
    if dry_run:
        args.append("--dry-run")
    monkeypatch.setattr(sys, "argv", args)
    with (
        patch.object(
            select,
            "list_pypi_distribution_names",
            return_value=["existing-project", "new-project"],
        ),
        patch.object(
            select, "project_exists_on_pypi", side_effect=[True, False]
        ),
    ):
        assert select.main() == 0
    assert existing.exists() is dry_run
    assert missing.exists()
    assert (
        output.read_text()
        == "previous=value\nhas_new_projects=true\nmissing_distributions=new-project\n"
    )
    assert "Kept wheels: 1" in summary.read_text()


def test_missing_wheel_fails_without_publication_outputs(
    tmp_path, monkeypatch
):
    output = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select",
            "--dist-dir",
            str(tmp_path),
            "--github-output",
            str(output),
        ],
    )
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    with (
        patch.object(
            select,
            "list_pypi_distribution_names",
            return_value=["new-project"],
        ),
        patch.object(select, "project_exists_on_pypi", return_value=False),
    ):
        with pytest.raises(RuntimeError, match="no matching wheel"):
            select.main()
    assert not output.exists()
