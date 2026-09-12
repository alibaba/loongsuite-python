from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "detect_loongsuite_changes.py"
SPEC = importlib.util.spec_from_file_location(
    "detect_loongsuite_changes",
    SCRIPT_PATH,
)
detect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detect)


def _run_detector(
    monkeypatch,
    tmp_path,
    changed_files=None,
    event=None,
    known_packages=None,
    tox_diff=None,
    event_name="pull_request",
):
    output_path = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_EVENT_NAME", event_name)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    if changed_files is None:
        monkeypatch.delenv("LOONGSUITE_CHANGED_FILES", raising=False)
    else:
        monkeypatch.setenv(
            "LOONGSUITE_CHANGED_FILES",
            "\n".join(changed_files),
        )

    if event is not None:
        event_path = tmp_path / "event.json"
        event_path.write_text(json.dumps(event), encoding="utf-8")
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    else:
        monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)

    if known_packages is not None:
        monkeypatch.setattr(
            detect,
            "_known_loongsuite_packages",
            lambda: set(known_packages),
        )

    if tox_diff is None:
        monkeypatch.delenv("LOONGSUITE_TOX_DIFF", raising=False)
    else:
        monkeypatch.setenv("LOONGSUITE_TOX_DIFF", tox_diff)

    assert detect.main() == 0

    return dict(
        line.split("=", 1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    )


def test_scopes_single_package_change(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [
            "instrumentation-loongsuite/"
            "loongsuite-instrumentation-crewai/src/package.py",
            "instrumentation-loongsuite/"
            "loongsuite-instrumentation-crewai/tests/test_package.py",
        ],
    )

    assert outputs["full"] == "false"
    assert outputs["packages"] == "|loongsuite-instrumentation-crewai|"
    assert outputs["degraded"] == "false"


def test_util_genai_change_runs_full_suite(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        ["util/opentelemetry-util-genai/src/opentelemetry/util/genai/foo.py"],
    )

    assert outputs["full"] == "true"
    assert outputs["degraded"] == "false"


def test_push_event_runs_full_suite(monkeypatch, tmp_path):
    monkeypatch.setattr(
        detect,
        "_run_git_diff",
        lambda base_ref: (_ for _ in ()).throw(
            AssertionError("push events should not diff against a PR base")
        ),
    )

    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        event_name="push",
    )

    assert outputs["full"] == "true"
    assert outputs["packages"] == "||"
    assert outputs["degraded"] == "false"
    assert outputs["reason"] == "non-pull-request event"


def test_merge_group_event_runs_full_suite(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        event_name="merge_group",
    )

    assert outputs["full"] == "true"
    assert outputs["packages"] == "||"
    assert outputs["degraded"] == "false"
    assert outputs["reason"] == "non-pull-request event"


def test_release_label_runs_full_suite(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [
            "instrumentation-loongsuite/loongsuite-instrumentation-crewai/foo.py"
        ],
        {"pull_request": {"labels": [{"name": "prepare-release"}]}},
    )

    assert outputs["full"] == "true"
    assert outputs["reason"] == "release or backport pull request"


def test_unknown_package_falls_back_to_full(monkeypatch, tmp_path, capsys):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [
            "instrumentation-loongsuite/"
            "loongsuite-instrumentation-newpkg/src/package.py"
        ],
    )

    captured = capsys.readouterr()
    assert outputs["full"] == "true"
    assert "loongsuite-instrumentation-newpkg" in outputs["reason"]
    assert "::error::" in captured.err


def test_new_plugin_with_generated_workflows_is_package_scoped(
    monkeypatch,
    tmp_path,
):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [
            "instrumentation-loongsuite/"
            "loongsuite-instrumentation-deepagents/src/package.py",
            "instrumentation-loongsuite/"
            "loongsuite-instrumentation-deepagents/tests/test_package.py",
            "instrumentation-loongsuite/"
            "loongsuite-instrumentation-langchain/src/package.py",
            "tox-loongsuite.ini",
            ".github/workflows/loongsuite_test_0.yml",
            ".github/workflows/loongsuite_lint_0.yml",
        ],
        {"pull_request": {"base": {"sha": "base-sha"}}},
        {
            "loongsuite-instrumentation-deepagents",
            "loongsuite-instrumentation-langchain",
        },
        "\n".join(
            [
                "+    ; loongsuite-instrumentation-deepagents",
                "+    py3{10,11,12,13}-test-"
                "loongsuite-instrumentation-deepagents",
                "+    lint-loongsuite-instrumentation-deepagents",
                "+  deepagents: -r {toxinidir}/instrumentation-loongsuite/"
                "loongsuite-instrumentation-deepagents/tests/"
                "test-requirements.txt",
                "+  test-loongsuite-instrumentation-deepagents: pytest "
                "{toxinidir}/instrumentation-loongsuite/"
                "loongsuite-instrumentation-deepagents/tests {posargs}",
            ]
        ),
    )

    assert outputs["full"] == "false"
    assert outputs["packages"] == (
        "|loongsuite-instrumentation-deepagents|"
        "loongsuite-instrumentation-langchain|"
    )


def test_generated_workflow_only_change_skips_jobs(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [".github/workflows/loongsuite_test_0.yml"],
    )

    assert outputs["full"] == "false"
    assert outputs["packages"] == "||"


def test_bootstrap_registry_fragment_scopes_package_change(
    monkeypatch,
    tmp_path,
):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [
            "loongsuite-distro/src/loongsuite/distro/bootstrap_registry/"
            "loongsuite_instrumentation_crewai.py"
        ],
        known_packages={"loongsuite-instrumentation-crewai"},
    )

    assert outputs["full"] == "false"
    assert outputs["packages"] == "|loongsuite-instrumentation-crewai|"


def test_bootstrap_registry_loader_change_runs_full_suite(
    monkeypatch,
    tmp_path,
):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [
            "loongsuite-distro/src/loongsuite/distro/bootstrap_registry/"
            "__init__.py"
        ],
    )

    assert outputs["full"] == "true"
    assert outputs["reason"] == (
        "shared LoongSuite file changed: "
        "loongsuite-distro/src/loongsuite/distro/bootstrap_registry/"
        "__init__.py"
    )


def test_upstream_bootstrap_registry_fragment_runs_full_suite(
    monkeypatch,
    tmp_path,
):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [
            "loongsuite-distro/src/loongsuite/distro/bootstrap_registry/"
            "opentelemetry_instrumentation_aiohttp_client.py"
        ],
    )

    assert outputs["full"] == "true"
    assert outputs["reason"] == (
        "shared LoongSuite file changed: "
        "loongsuite-distro/src/loongsuite/distro/bootstrap_registry/"
        "opentelemetry_instrumentation_aiohttp_client.py"
    )


def test_release_workflow_change_runs_full_suite(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [".github/workflows/loongsuite-release.yml"],
    )

    assert outputs["full"] == "true"
    assert (
        outputs["reason"] == "shared LoongSuite file changed: "
        ".github/workflows/loongsuite-release.yml"
    )


def test_matrix_selector_change_runs_full_suite(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [".github/scripts/select_loongsuite_matrix.py"],
    )

    assert outputs["full"] == "true"
    assert (
        outputs["reason"] == "shared LoongSuite file changed: "
        ".github/scripts/select_loongsuite_matrix.py"
    )


def test_generate_workflows_lib_metadata_change_runs_full_suite(
    monkeypatch,
    tmp_path,
):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [".github/workflows/generate_workflows_lib/pyproject.toml"],
    )

    assert outputs["full"] == "true"
    assert outputs["reason"] == (
        "shared LoongSuite file changed: "
        ".github/workflows/generate_workflows_lib/pyproject.toml"
    )


def test_generate_workflows_lib_template_change_runs_full_suite(
    monkeypatch,
    tmp_path,
):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [
            ".github/workflows/generate_workflows_lib/src/"
            "generate_workflows_lib/loongsuite_lint.yml.j2"
        ],
    )

    assert outputs["full"] == "true"


def test_generate_workflows_lib_test_change_runs_full_suite(
    monkeypatch,
    tmp_path,
):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [".github/workflows/generate_workflows_lib/tests/test_rendering.py"],
    )

    assert outputs["full"] == "true"


def test_unknown_loongsuite_workflow_change_runs_full_suite(
    monkeypatch,
    tmp_path,
):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [".github/workflows/loongsuite_other_0.yml"],
    )

    assert outputs["full"] == "true"


def test_generated_workflow_name_matching():
    assert detect._is_generated_loongsuite_workflow(
        ".github/workflows/loongsuite_test_10.yml"
    )
    assert not detect._is_generated_loongsuite_workflow(
        ".github/workflows/loongsuite_other_0.yml"
    )
    assert not detect._is_generated_loongsuite_workflow(
        ".github/workflows/loongsuite_test_0.yaml"
    )
    assert not detect._is_generated_loongsuite_workflow(
        ".github/workflows/loongsuite-test_0.yml"
    )


def test_tox_only_scoped_change_runs_package_jobs(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        ["tox-loongsuite.ini"],
        {"pull_request": {"base": {"sha": "base-sha"}}},
        {"loongsuite-instrumentation-deepagents"},
        "\n".join(
            [
                "+    py3{10,11,12,13}-test-"
                "loongsuite-instrumentation-deepagents",
                "+    lint-loongsuite-instrumentation-deepagents",
            ]
        ),
    )

    assert outputs["full"] == "false"
    assert outputs["packages"] == "|loongsuite-instrumentation-deepagents|"


def test_shared_tox_change_runs_full_suite(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        ["tox-loongsuite.ini"],
        {"pull_request": {"base": {"sha": "base-sha"}}},
        {"loongsuite-instrumentation-crewai"},
        "+  CORE_REPO_SHA={env:CORE_REPO_SHA:main}",
    )

    assert outputs["full"] == "true"
    assert outputs["reason"].startswith("shared tox-loongsuite.ini change:")


def test_util_genai_tox_change_runs_full_suite(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        ["tox-loongsuite.ini"],
        {"pull_request": {"base": {"sha": "base-sha"}}},
        {"util-genai"},
        "+    py3{10,11,12,13}-test-util-genai",
    )

    assert outputs["full"] == "true"
    assert outputs["reason"].startswith("shared tox-loongsuite.ini change:")


def test_package_mentions_do_not_match_package_prefixes():
    assert detect._packages_from_text(
        "lint-loongsuite-instrumentation-langchain-core",
        {
            "loongsuite-instrumentation-langchain",
            "loongsuite-instrumentation-langchain-core",
        },
    ) == {"loongsuite-instrumentation-langchain-core"}


def test_changed_tox_lines_ignores_headers_comments_and_blanks():
    diff_text = "\n".join(
        [
            "diff --git a/tox-loongsuite.ini b/tox-loongsuite.ini",
            "--- a/tox-loongsuite.ini",
            "+++ b/tox-loongsuite.ini",
            "@@ -1,0 +1,2 @@",
            "+",
            "+    ; loongsuite-instrumentation-deepagents",
            "+    lint-loongsuite-instrumentation-deepagents",
            "-    lint-loongsuite-instrumentation-langchain",
        ]
    )

    assert detect._changed_tox_lines(diff_text) == [
        "lint-loongsuite-instrumentation-deepagents",
        "lint-loongsuite-instrumentation-langchain",
    ]


def test_empty_changed_files_is_degraded_full_run(monkeypatch, tmp_path):
    outputs = _run_detector(monkeypatch, tmp_path, [])

    assert outputs["full"] == "true"
    assert outputs["degraded"] == "true"
    assert outputs["reason"] == "unable to compute changed files"


def test_package_docs_only_change_skips_package(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [
            "instrumentation-loongsuite/"
            "loongsuite-instrumentation-crewai/README.md",
            "instrumentation-loongsuite/"
            "loongsuite-instrumentation-crewai/CHANGELOG.md",
        ],
    )

    assert outputs["full"] == "false"
    assert outputs["packages"] == "||"


def test_unexpected_detector_error_falls_back_to_full(monkeypatch, tmp_path):
    def broken_changed_files(_event):
        raise KeyError("boom")

    monkeypatch.setattr(detect, "_changed_files", broken_changed_files)
    outputs = _run_detector(monkeypatch, tmp_path, None)

    assert outputs["full"] == "true"
    assert outputs["degraded"] == "true"
    assert "unexpected detector failure" in outputs["reason"]
