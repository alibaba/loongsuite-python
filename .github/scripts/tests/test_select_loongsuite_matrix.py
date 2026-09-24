from __future__ import annotations

import importlib.util
import json
from json import JSONDecodeError
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "select_loongsuite_matrix.py"
SPEC = importlib.util.spec_from_file_location(
    "select_loongsuite_matrix",
    SCRIPT_PATH,
)
select = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(select)


ALL_JOBS = [
    {
        "package": "loongsuite-instrumentation-google-adk",
        "tox_env": "lint-loongsuite-instrumentation-google-adk",
    },
    {
        "package": "loongsuite-instrumentation-langchain",
        "tox_env": "lint-loongsuite-instrumentation-langchain",
    },
]


def test_full_run_selects_all_jobs():
    assert (
        select.select_jobs(
            ALL_JOBS,
            full=True,
            package_output="||",
        )
        == ALL_JOBS
    )


def test_package_scoped_run_selects_matching_jobs():
    selected_jobs = select.select_jobs(
        ALL_JOBS,
        full=False,
        package_output="|loongsuite-instrumentation-langchain|",
    )

    assert selected_jobs == [ALL_JOBS[1]]


def test_no_package_change_selects_no_jobs():
    assert (
        select.select_jobs(
            ALL_JOBS,
            full=False,
            package_output="||",
        )
        == []
    )


def test_malformed_package_entries_are_ignored():
    selected_jobs = select.select_jobs(
        [
            {"tox_env": "missing-package"},
            {"package": None, "tox_env": "none-package"},
            {"package": "loongsuite-instrumentation-langchain"},
        ],
        full=False,
        package_output="|loongsuite-instrumentation-langchain|",
    )

    assert selected_jobs == [
        {"package": "loongsuite-instrumentation-langchain"}
    ]


def test_main_writes_dynamic_matrix_outputs(monkeypatch, tmp_path):
    output_path = tmp_path / "github-output"
    monkeypatch.setenv("LOONGSUITE_ALL_JOBS", json.dumps(ALL_JOBS))
    monkeypatch.setenv("LOONGSUITE_FULL", "false")
    monkeypatch.setenv(
        "LOONGSUITE_PACKAGES",
        "|loongsuite-instrumentation-google-adk|",
    )
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert select.main() == 0

    outputs = dict(
        line.split("=", 1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    )
    assert outputs["has_jobs"] == "true"
    assert json.loads(outputs["matrix"]) == {"include": [ALL_JOBS[0]]}


def test_main_full_run_includes_all_jobs(monkeypatch, tmp_path):
    output_path = tmp_path / "github-output"
    monkeypatch.setenv("LOONGSUITE_ALL_JOBS", json.dumps(ALL_JOBS))
    monkeypatch.setenv("LOONGSUITE_FULL", "true")
    monkeypatch.delenv("LOONGSUITE_PACKAGES", raising=False)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert select.main() == 0

    outputs = dict(
        line.split("=", 1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    )
    assert outputs["has_jobs"] == "true"
    assert json.loads(outputs["matrix"]) == {"include": ALL_JOBS}


def test_main_no_package_changes_writes_empty_matrix(monkeypatch, tmp_path):
    output_path = tmp_path / "github-output"
    monkeypatch.setenv("LOONGSUITE_ALL_JOBS", json.dumps(ALL_JOBS))
    monkeypatch.setenv("LOONGSUITE_FULL", "false")
    monkeypatch.setenv("LOONGSUITE_PACKAGES", "||")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert select.main() == 0

    outputs = dict(
        line.split("=", 1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    )
    assert outputs["has_jobs"] == "false"
    assert json.loads(outputs["matrix"]) == {"include": []}


def test_main_full_mode_requires_literal_true(monkeypatch, tmp_path):
    output_path = tmp_path / "github-output"
    monkeypatch.setenv("LOONGSUITE_ALL_JOBS", json.dumps(ALL_JOBS))
    monkeypatch.setenv("LOONGSUITE_FULL", "True")
    monkeypatch.setenv("LOONGSUITE_PACKAGES", "||")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert select.main() == 0

    outputs = dict(
        line.split("=", 1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    )
    assert outputs["has_jobs"] == "false"
    assert json.loads(outputs["matrix"]) == {"include": []}


def test_load_all_jobs_requires_env_var(monkeypatch):
    monkeypatch.delenv("LOONGSUITE_ALL_JOBS", raising=False)

    with pytest.raises(ValueError, match="LOONGSUITE_ALL_JOBS is required"):
        select._load_all_jobs()


def test_load_all_jobs_rejects_non_list(monkeypatch):
    monkeypatch.setenv("LOONGSUITE_ALL_JOBS", "{}")

    with pytest.raises(ValueError, match="must be a JSON list"):
        select._load_all_jobs()


def test_load_all_jobs_rejects_empty_list(monkeypatch):
    monkeypatch.setenv("LOONGSUITE_ALL_JOBS", "[]")

    with pytest.raises(ValueError, match="is empty"):
        select._load_all_jobs()


def test_main_rejects_malformed_all_jobs(monkeypatch, capsys):
    monkeypatch.setenv("LOONGSUITE_ALL_JOBS", "not json")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    with pytest.raises(JSONDecodeError):
        select.main()

    assert (
        "::error::LoongSuite matrix selector failed" in capsys.readouterr().err
    )


@pytest.mark.parametrize("raw_jobs", ["{}", "[]"])
def test_main_rejects_invalid_all_jobs_shape(monkeypatch, capsys, raw_jobs):
    monkeypatch.setenv("LOONGSUITE_ALL_JOBS", raw_jobs)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    with pytest.raises(ValueError):
        select.main()

    assert (
        "::error::LoongSuite matrix selector failed" in capsys.readouterr().err
    )


def test_main_allows_missing_github_output(monkeypatch):
    monkeypatch.setenv("LOONGSUITE_ALL_JOBS", json.dumps(ALL_JOBS))
    monkeypatch.setenv("LOONGSUITE_FULL", "true")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    assert select.main() == 0
