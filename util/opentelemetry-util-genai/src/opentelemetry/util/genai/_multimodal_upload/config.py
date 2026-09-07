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

"""Runtime configuration snapshot for multimodal upload."""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Tuple, Union

from opentelemetry.util.genai.extended_environment_variables import (  # pylint: disable=no-name-in-module
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_ALLOWED_ROOT_PATHS,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_LOCAL_FILE_ENABLED,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_BUCKET,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_PATH_PREFIX,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_ENDPOINT,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_LICENSE_KEY,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_WORKSPACE,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER,
)

_logger = logging.getLogger(__name__)

_VALID_UPLOAD_MODES = frozenset({"none", "input", "output", "both"})
_VALID_HOOKS = frozenset({"arms", "fs", "oss", "presign"})
PRESIGN_HOOK_NAME = "presign"
# Console/ConfigServer aliases for the pre-authorized OSS mode.
_HOOK_NAME_ALIASES = {
    "oss-presign": PRESIGN_HOOK_NAME,
    "oss_presign": PRESIGN_HOOK_NAME,
    "presigned-oss": PRESIGN_HOOK_NAME,
    "presigned_oss": PRESIGN_HOOK_NAME,
}
DEFAULT_SLS_LOGSTORE = "logstore-multimodal"
DEFAULT_MULTIMODAL_UPLOADER_HOOK = "fs"
DEFAULT_MULTIMODAL_PRE_UPLOADER_HOOK = "fs"

# Optional env for bootstrap / tests (also used by commercial SLS hooks).
_APSARA_SLS_PROJECT_ENV = "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_PROJECT"
_APSARA_SLS_LOGSTORE_ENV = "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_LOGSTORE"
_APSARA_SLS_ENDPOINT_ENV = "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_ENDPOINT"
_APSARA_SLS_AUTH_TYPE_ENV = "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_AUTH_TYPE"
_APSARA_SLS_ACCESS_KEY_ID_ENV = (
    "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_ACCESS_KEY_ID"
)
_APSARA_SLS_ACCESS_KEY_SECRET_ENV = (
    "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_ACCESS_KEY_SECRET"
)
_APSARA_SLS_STS_TOKEN_ENV = "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_STS_TOKEN"

# Identity authenticating presign requests. The dedicated multimodal
# variables take precedence so an application can upload without running
# under an ARMS agent; the ARMS_* ones remain as a fallback so a
# co-located agent needs no extra configuration.
_ARMS_LICENSE_KEY_ENV = "ARMS_LICENSE_KEY"
_ARMS_WORKSPACE_ENV = "ARMS_WORKSPACE"

STRATEGY_FIELDS = frozenset(
    {
        "upload_mode",
        "download_enabled",
        "local_file_enabled",
        "allowed_root_paths",
    }
)

UPLOADER_GENERATION_FIELDS = frozenset(
    {
        "storage_base_path",
        "uploader_hook_name",
        "pre_uploader_hook_name",
        "sls_project",
        "sls_logstore",
        "oss_bucket",
        "oss_path_prefix",
    }
)

# SLS connection/credential fields are env-only for security: runtime updates must
# not rotate credentials or endpoints through update_multimodal_runtime_config().
_READ_ONLY_RUNTIME_FIELDS = frozenset(
    {
        "sls_endpoint",
        "sls_auth_type",
        "sls_access_key_id",
        "sls_access_key_secret",
        "sls_sts_token",
        "presign_endpoint",
        "presign_license_key",
        "presign_workspace",
    }
)


@dataclass(frozen=True)
class MultimodalConfigSnapshot:
    upload_mode: str
    download_enabled: bool
    local_file_enabled: bool
    allowed_root_paths: Tuple[str, ...]
    storage_base_path: Optional[str]
    uploader_hook_name: str
    pre_uploader_hook_name: str
    sls_project: Optional[str]
    sls_logstore: Optional[str]
    sls_endpoint: Optional[str]
    sls_auth_type: Optional[str]
    sls_access_key_id: Optional[str]
    sls_access_key_secret: Optional[str]
    sls_sts_token: Optional[str]
    oss_bucket: Optional[str] = None
    oss_path_prefix: Optional[str] = None
    presign_endpoint: Optional[str] = None
    presign_license_key: Optional[str] = None
    presign_workspace: Optional[str] = None
    version: int = 0
    strategy_version: int = 0
    uploader_generation: int = 0

    @property
    def process_input(self) -> bool:
        return self.upload_mode in ("input", "both")

    @property
    def process_output(self) -> bool:
        return self.upload_mode in ("output", "both")

    @property
    def effective_storage_base_path(self) -> Optional[str]:
        if self.uploader_hook_name == "arms":
            return format_sls_base_path(self.sls_project, self.sls_logstore)
        if self.uploader_hook_name == PRESIGN_HOOK_NAME:
            return format_sls_base_path(
                self.sls_project,
                self.sls_logstore,
                self.oss_path_prefix,
            )
        return self.storage_base_path


def _parse_env_bool(value: Optional[str], default: bool = False) -> bool:
    if not value:
        return default
    return value.lower() in ("true", "1", "yes")


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip()
    if not text:
        return default
    return text.lower() in ("true", "1", "yes")


def _normalize_upload_mode(value: Optional[str]) -> str:
    if not value:
        return "none"
    mode = str(value).strip().lower()
    if mode not in _VALID_UPLOAD_MODES:
        _logger.warning(
            "Invalid multimodal upload_mode %r, fallback to none", value
        )
        return "none"
    return mode


def normalize_multimodal_hook_name(value: Any) -> Optional[str]:
    """Return normalized hook name, or None if unsupported."""
    if value is None:
        return None
    hook_name = str(value).strip().lower()
    hook_name = _HOOK_NAME_ALIASES.get(hook_name, hook_name)
    if hook_name not in _VALID_HOOKS:
        _logger.warning("Unsupported multimodal hook name: %r", value)
        return None
    return hook_name


def normalize_oss_bucket(value: Any) -> Optional[str]:
    """Return ``bucket`` or ``bucket/prefix`` from a bucket name or oss:// URL."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "://" in text:
        scheme, _, remainder = text.partition("://")
        if scheme.lower() != "oss":
            _logger.warning(
                "Multimodal OSS bucket must be a name or oss:// URL: %r", value
            )
            return None
        text = remainder
    text = text.strip("/")
    if not text:
        return None
    if any(part in ("", ".", "..") for part in text.split("/")):
        _logger.warning(
            "Multimodal OSS bucket must not contain empty or dot segments: %r",
            value,
        )
        return None
    return text


def normalize_oss_path_prefix(value: Any) -> Optional[str]:
    """Return a clean ``a/b`` object path prefix, or None when unusable."""
    if value is None:
        return None
    text = str(value).strip().strip("/")
    if not text:
        return None
    if any(part in ("", ".", "..") for part in text.split("/")):
        _logger.warning(
            "Multimodal OSS path prefix must not contain empty or dot "
            "segments: %r",
            value,
        )
        return None
    return text


def format_sls_base_path(
    project: Optional[str],
    logstore: Optional[str],
    prefix: Optional[str] = None,
) -> Optional[str]:
    """Return the ``sls://{project}/{logstore}[/{prefix}]`` base path.

    Both the ARMS SLS uploader and the pre-authorized OSS uploader address
    objects this way: the backing bucket belongs to the server, so only
    project, logstore and object name identify an object. Returns None when no
    project is known or either target field is invalid, since the address
    would then be ambiguous.
    """
    project_name = (project or "").strip().strip("/")
    if not project_name:
        return None
    store = (logstore or "").strip().strip("/") or DEFAULT_SLS_LOGSTORE
    for field_name, value in (("project", project_name), ("logstore", store)):
        if "/" in value or value in (".", ".."):
            _logger.warning(
                "Multimodal SLS %s must not contain slashes or dot segments: %r",
                field_name,
                value,
            )
            return None
    base_path = f"sls://{project_name}/{store}"
    normalized_prefix = normalize_oss_path_prefix(prefix)
    if normalized_prefix:
        return f"{base_path}/{normalized_prefix}"
    return base_path


def _first_env(*names: str) -> Optional[str]:
    """Return the first non-empty value among ``names``."""
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def _normalize_allowed_root_paths(
    value: Union[str, Sequence[str], None],
) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        if not value.strip():
            return ()
        parts = [p.strip() for p in re.split(r"[,]", value) if p.strip()]
    else:
        parts = [str(p).strip() for p in value if str(p).strip()]
    normalized: list[str] = []
    seen: set[str] = set()
    for part in parts:
        abs_path = os.path.abspath(part)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        if not os.path.exists(abs_path):
            _logger.debug(
                "Multimodal allowed_root_paths entry does not exist: %s",
                abs_path,
            )
        normalized.append(abs_path)
    return tuple(normalized)


def _snapshot_from_env(*, version: int = 0) -> MultimodalConfigSnapshot:
    upload_mode = _normalize_upload_mode(
        os.getenv(OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE, "none")
    )
    download_enabled = _parse_env_bool(
        os.getenv(OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED)
    )
    local_file_enabled = _parse_env_bool(
        os.getenv(OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_LOCAL_FILE_ENABLED)
    )
    allowed_root_paths = _normalize_allowed_root_paths(
        os.getenv(OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_ALLOWED_ROOT_PATHS, "")
    )
    storage_base_path = os.getenv(
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH
    )
    uploader_hook_name = (
        normalize_multimodal_hook_name(
            os.getenv(
                OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER,
                DEFAULT_MULTIMODAL_UPLOADER_HOOK,
            )
        )
        or DEFAULT_MULTIMODAL_UPLOADER_HOOK
    )
    pre_uploader_hook_name = (
        normalize_multimodal_hook_name(
            os.getenv(
                OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER,
                DEFAULT_MULTIMODAL_PRE_UPLOADER_HOOK,
            )
        )
        or DEFAULT_MULTIMODAL_PRE_UPLOADER_HOOK
    )

    return MultimodalConfigSnapshot(
        upload_mode=upload_mode,
        download_enabled=download_enabled,
        local_file_enabled=local_file_enabled,
        allowed_root_paths=allowed_root_paths,
        storage_base_path=storage_base_path,
        uploader_hook_name=uploader_hook_name,
        pre_uploader_hook_name=pre_uploader_hook_name,
        sls_project=os.getenv(_APSARA_SLS_PROJECT_ENV) or None,
        sls_logstore=os.getenv(_APSARA_SLS_LOGSTORE_ENV) or None,
        sls_endpoint=os.getenv(_APSARA_SLS_ENDPOINT_ENV) or None,
        sls_auth_type=os.getenv(_APSARA_SLS_AUTH_TYPE_ENV) or None,
        sls_access_key_id=os.getenv(_APSARA_SLS_ACCESS_KEY_ID_ENV) or None,
        sls_access_key_secret=os.getenv(_APSARA_SLS_ACCESS_KEY_SECRET_ENV)
        or None,
        sls_sts_token=os.getenv(_APSARA_SLS_STS_TOKEN_ENV) or None,
        oss_bucket=normalize_oss_bucket(
            os.getenv(OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_BUCKET)
        ),
        oss_path_prefix=normalize_oss_path_prefix(
            os.getenv(OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_PATH_PREFIX)
        ),
        presign_endpoint=os.getenv(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_ENDPOINT
        )
        or None,
        presign_license_key=_first_env(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_LICENSE_KEY,
            _ARMS_LICENSE_KEY_ENV,
        ),
        presign_workspace=_first_env(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_WORKSPACE,
            _ARMS_WORKSPACE_ENV,
        ),
        version=version,
        strategy_version=0,
        uploader_generation=0,
    )


def _coalesce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_and_validate(
    old: MultimodalConfigSnapshot,
    fields: dict[str, Any],
) -> MultimodalConfigSnapshot:
    merged: dict[str, Any] = {
        "upload_mode": old.upload_mode,
        "download_enabled": old.download_enabled,
        "local_file_enabled": old.local_file_enabled,
        "allowed_root_paths": old.allowed_root_paths,
        "storage_base_path": old.storage_base_path,
        "uploader_hook_name": old.uploader_hook_name,
        "pre_uploader_hook_name": old.pre_uploader_hook_name,
        "sls_project": old.sls_project,
        "sls_logstore": old.sls_logstore,
        "sls_endpoint": old.sls_endpoint,
        "sls_auth_type": old.sls_auth_type,
        "sls_access_key_id": old.sls_access_key_id,
        "sls_access_key_secret": old.sls_access_key_secret,
        "sls_sts_token": old.sls_sts_token,
        "oss_bucket": old.oss_bucket,
        "oss_path_prefix": old.oss_path_prefix,
    }

    for key, value in fields.items():
        if value is None:
            continue
        if key in _READ_ONLY_RUNTIME_FIELDS:
            _logger.warning(
                "Ignoring runtime update for read-only multimodal field %r "
                "(SLS credentials/endpoint must come from environment)",
                key,
            )
            continue
        if key not in merged:
            continue
        if key == "upload_mode":
            merged[key] = _normalize_upload_mode(value)
        elif key in ("download_enabled", "local_file_enabled"):
            merged[key] = _coerce_bool(value)
        elif key == "allowed_root_paths":
            merged[key] = _normalize_allowed_root_paths(value)
        elif key in ("uploader_hook_name", "pre_uploader_hook_name"):
            normalized_hook = normalize_multimodal_hook_name(value)
            if normalized_hook is not None:
                merged[key] = normalized_hook
        elif key == "oss_bucket":
            merged[key] = normalize_oss_bucket(value)
        elif key == "oss_path_prefix":
            merged[key] = normalize_oss_path_prefix(value)
        elif key in (
            "storage_base_path",
            "sls_project",
            "sls_logstore",
        ):
            merged[key] = _coalesce_optional_str(value)

    return MultimodalConfigSnapshot(
        upload_mode=merged["upload_mode"],
        download_enabled=merged["download_enabled"],
        local_file_enabled=merged["local_file_enabled"],
        allowed_root_paths=merged["allowed_root_paths"],
        storage_base_path=merged["storage_base_path"],
        uploader_hook_name=merged["uploader_hook_name"],
        pre_uploader_hook_name=merged["pre_uploader_hook_name"],
        sls_project=merged["sls_project"],
        sls_logstore=merged["sls_logstore"],
        sls_endpoint=merged["sls_endpoint"],
        sls_auth_type=merged["sls_auth_type"],
        sls_access_key_id=merged["sls_access_key_id"],
        sls_access_key_secret=merged["sls_access_key_secret"],
        sls_sts_token=merged["sls_sts_token"],
        oss_bucket=merged["oss_bucket"],
        oss_path_prefix=merged["oss_path_prefix"],
        presign_endpoint=old.presign_endpoint,
        presign_license_key=old.presign_license_key,
        presign_workspace=old.presign_workspace,
        version=old.version,
        strategy_version=old.strategy_version,
        uploader_generation=old.uploader_generation,
    )


def _next_strategy_version(
    old: MultimodalConfigSnapshot,
    new: MultimodalConfigSnapshot,
) -> int:
    if any(
        getattr(old, field) != getattr(new, field) for field in STRATEGY_FIELDS
    ):
        return old.strategy_version + 1
    return old.strategy_version


def _next_uploader_generation(
    old: MultimodalConfigSnapshot,
    new: MultimodalConfigSnapshot,
) -> int:
    if any(
        getattr(old, field) != getattr(new, field)
        for field in UPLOADER_GENERATION_FIELDS
    ):
        return old.uploader_generation + 1
    return old.uploader_generation


class MultimodalRuntimeConfig:
    """Process-wide mutable multimodal runtime configuration."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot = _snapshot_from_env(version=0)

    def get_snapshot(self) -> MultimodalConfigSnapshot:
        return self._snapshot

    def update(self, **fields: Any) -> MultimodalConfigSnapshot:
        with self._lock:
            old = self._snapshot
            new = _normalize_and_validate(old, fields)
            if new == old:
                return self._snapshot
            new = replace(
                new,
                version=old.version + 1,
                strategy_version=_next_strategy_version(old, new),
                uploader_generation=_next_uploader_generation(old, new),
            )
            self._snapshot = new
            return self._snapshot


_runtime_config = MultimodalRuntimeConfig()


def get_multimodal_config_snapshot() -> MultimodalConfigSnapshot:
    return _runtime_config.get_snapshot()


def update_multimodal_runtime_config(
    **fields: Any,
) -> MultimodalConfigSnapshot:
    return _runtime_config.update(**fields)
