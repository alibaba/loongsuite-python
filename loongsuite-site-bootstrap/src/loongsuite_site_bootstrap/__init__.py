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
LoongSuite site-packages bootstrap: imported from a .pth line during site
initialization when this distribution is installed.

Gate for ``LOONGSUITE_PYTHON_SITE_BOOTSTRAP`` (string ``true``, case-insensitive
after stripping whitespace; anything else is off):

1. **Set and not ``true``** — exit immediately: do **not** read
   ``bootstrap-config.json``, do not run OpenTelemetry.
2. **Unset** — read ``~/.loongsuite/bootstrap-config.json`` (if present); enable
   only when that file maps the key to ``true``. Otherwise skip.
3. **Set and ``true``** — still read ``bootstrap-config.json`` when the file
   exists, so JSON can **fill in** other keys that are missing from
   ``os.environ``; OpenTelemetry runs because the env switch is on.

When enabled, keys from ``bootstrap-config.json`` are applied only for names
**missing** from ``os.environ`` (``setdefault``-like semantics), then optional
OpenTelemetry auto-instrumentation runs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from loongsuite_site_bootstrap.version import __version__

LOONGSUITE_PYTHON_SITE_BOOTSTRAP = "LOONGSUITE_PYTHON_SITE_BOOTSTRAP"
LOONGSUITE_PYTHON_SITE_BOOTSTRAP_LOG_SUCCESS = (
    "LOONGSUITE_PYTHON_SITE_BOOTSTRAP_LOG_SUCCESS"
)
LOONGSUITE_PYTHON_SITE_BOOTSTRAP_STATUS_FILE = (
    "LOONGSUITE_PYTHON_SITE_BOOTSTRAP_STATUS_FILE"
)
_LOGGER: logging.Logger = logging.getLogger(__name__)
_INITIALIZED = False


def _configure_bootstrap_logging() -> None:
    """Emit bootstrap messages to stdout even before the app configures logging."""
    if _LOGGER.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


_configure_bootstrap_logging()


def _coerce_env_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _is_truthy_string(val: str) -> bool:
    """True only for case-insensitive ``true``; any other string is off."""
    return val.strip().lower() == "true"


def _is_falsey_string(val: str) -> bool:
    """Return True for common explicit-off strings."""
    return val.strip().lower() in {"false", "0", "no", "off"}


def _should_log_success() -> bool:
    val = os.environ.get(LOONGSUITE_PYTHON_SITE_BOOTSTRAP_LOG_SUCCESS)
    if val is None:
        return True
    return not _is_falsey_string(val)


def _write_status_file(initialized: bool, error: str | None = None) -> None:
    path_value = os.environ.get(LOONGSUITE_PYTHON_SITE_BOOTSTRAP_STATUS_FILE)
    if not path_value:
        return
    path = Path(path_value).expanduser()
    payload: dict[str, object] = {
        "initialized": initialized,
        "pid": os.getpid(),
        "version": __version__,
    }
    if error:
        payload["error"] = error
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(
            json.dumps(payload, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    except Exception:
        _LOGGER.debug(
            "loongsuite-site-bootstrap: failed to write status file %s",
            path,
            exc_info=True,
        )


def _read_bootstrap_config_file() -> dict[str, str] | None:
    path = Path.home() / ".loongsuite" / "bootstrap-config.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
    ) as exc:
        _LOGGER.warning(
            "Ignoring invalid LoongSuite bootstrap config %s: %s", path, exc
        )
        return None
    if not isinstance(data, dict):
        _LOGGER.warning(
            "Ignoring LoongSuite bootstrap config %s: root must be a JSON object",
            path,
        )
        return None
    file_defaults: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        coerced = _coerce_env_value(value)
        if coerced is None:
            continue
        file_defaults[key] = coerced
    return file_defaults if file_defaults else None


def _apply_bootstrap_config_defaults(
    file_defaults: dict[str, str] | None,
) -> None:
    if not file_defaults:
        _LOGGER.debug(
            "loongsuite-site-bootstrap: no bootstrap-config.json keys to consider "
            "(file missing, invalid, or empty)",
        )
        return
    applied: list[str] = []
    skipped: list[str] = []
    for key, value in file_defaults.items():
        if key in os.environ:
            skipped.append(key)
            continue
        os.environ[key] = value
        applied.append(key)
    _LOGGER.debug(
        "loongsuite-site-bootstrap: from bootstrap-config.json, "
        "set %d unset key(s): %s; skipped %d already set: %s",
        len(applied),
        ", ".join(applied) if applied else "(none)",
        len(skipped),
        ", ".join(skipped) if skipped else "(none)",
    )


def _bootstrap_switch_enabled(file_defaults: dict[str, str] | None) -> bool:
    """Return whether the bootstrap feature is enabled (without reading JSON twice)."""
    env_val = os.environ.get(LOONGSUITE_PYTHON_SITE_BOOTSTRAP)
    if env_val is not None:
        return _is_truthy_string(env_val)
    if not file_defaults:
        return False
    cfg_val = file_defaults.get(LOONGSUITE_PYTHON_SITE_BOOTSTRAP)
    if cfg_val is None:
        return False
    return _is_truthy_string(cfg_val)


def _run_bootstrap_if_enabled() -> None:
    env_val = os.environ.get(LOONGSUITE_PYTHON_SITE_BOOTSTRAP)
    if env_val is not None and not _is_truthy_string(env_val):
        return

    file_defaults = _read_bootstrap_config_file()
    if not _bootstrap_switch_enabled(file_defaults):
        return

    _apply_bootstrap_config_defaults(file_defaults)
    _run_auto_instrumentation()


def _run_auto_instrumentation() -> None:
    global _INITIALIZED  # noqa: PLW0603
    # Align with loongsuite-distro + opentelemetry-instrument / sitecustomize
    os.environ.setdefault("OTEL_PYTHON_DISTRO", "loongsuite")
    os.environ.setdefault("OTEL_PYTHON_CONFIGURATOR", "loongsuite")

    try:
        from opentelemetry.instrumentation.auto_instrumentation import (  # noqa: PLC0415
            initialize,
        )

        initialize()
    except Exception:
        _INITIALIZED = False
        _write_status_file(False, "OpenTelemetry auto-instrumentation failed")
        _LOGGER.exception(
            "loongsuite-site-bootstrap: OpenTelemetry auto-instrumentation failed "
            "(import or initialize); continuing without instrumentation. "
            "Fix deps or unset LOONGSUITE_PYTHON_SITE_BOOTSTRAP if unintended."
        )
        return

    _INITIALIZED = True
    os.environ["LOONGSUITE_PYTHON_SITE_BOOTSTRAP_STARTED"] = "true"
    _write_status_file(True)
    if _should_log_success():
        _LOGGER.info(
            "loongsuite-site-bootstrap: started successfully "
            "(OpenTelemetry auto-instrumentation initialized)."
        )


def is_initialized() -> bool:
    return _INITIALIZED


_run_bootstrap_if_enabled()

__all__ = [
    "LOONGSUITE_PYTHON_SITE_BOOTSTRAP",
    "LOONGSUITE_PYTHON_SITE_BOOTSTRAP_LOG_SUCCESS",
    "LOONGSUITE_PYTHON_SITE_BOOTSTRAP_STATUS_FILE",
    "__version__",
    "is_initialized",
]
