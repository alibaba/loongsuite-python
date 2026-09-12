import importlib
import json
import sys


def _load_bootstrap(monkeypatch):
    monkeypatch.setenv("LOONGSUITE_PYTHON_SITE_BOOTSTRAP", "false")
    sys.modules.pop("loongsuite_site_bootstrap", None)
    return importlib.import_module("loongsuite_site_bootstrap")


def test_write_status_file_uses_atomic_json_payload(tmp_path, monkeypatch):
    bootstrap = _load_bootstrap(monkeypatch)
    status_path = tmp_path / "loongsuite-site-bootstrap-status.json"
    monkeypatch.setenv(
        bootstrap.LOONGSUITE_PYTHON_SITE_BOOTSTRAP_STATUS_FILE,
        str(status_path),
    )

    bootstrap._write_status_file(True)

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload == {
        "initialized": True,
        "pid": payload["pid"],
        "version": bootstrap.__version__,
    }
    assert isinstance(payload["pid"], int)
    assert list(tmp_path.glob(".*.tmp")) == []


def test_should_log_success_defaults_on_and_accepts_common_falsey(
    monkeypatch,
):
    bootstrap = _load_bootstrap(monkeypatch)
    monkeypatch.delenv(
        bootstrap.LOONGSUITE_PYTHON_SITE_BOOTSTRAP_LOG_SUCCESS,
        raising=False,
    )
    assert bootstrap._should_log_success() is True

    for value in ("False", "0", "no", "off"):
        monkeypatch.setenv(
            bootstrap.LOONGSUITE_PYTHON_SITE_BOOTSTRAP_LOG_SUCCESS,
            value,
        )
        assert bootstrap._should_log_success() is False
