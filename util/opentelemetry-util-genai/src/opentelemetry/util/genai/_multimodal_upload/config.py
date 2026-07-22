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
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER,
)

_logger = logging.getLogger(__name__)

_VALID_UPLOAD_MODES = frozenset({"none", "input", "output", "both"})
_VALID_HOOKS = frozenset({"arms", "fs"})
DEFAULT_SLS_LOGSTORE = "logstore-multimodal"
DEFAULT_MULTIMODAL_UPLOADER_HOOK = "fs"
DEFAULT_MULTIMODAL_PRE_UPLOADER_HOOK = "fs"

# Optional env for bootstrap / tests (also used by commercial SLS hooks).
_APSARA_SLS_PROJECT_ENV = "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_PROJECT"
_APSARA_SLS_LOGSTORE_ENV = "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_LOGSTORE"
_APSARA_SLS_ENDPOINT_ENV = "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_ENDPOINT"
_APSARA_SLS_AUTH_TYPE_ENV = "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_AUTH_TYPE"
_APSARA_SLS_ACCESS_KEY_ID_ENV = "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_ACCESS_KEY_ID"
_APSARA_SLS_ACCESS_KEY_SECRET_ENV = (
    "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_ACCESS_KEY_SECRET"
)
_APSARA_SLS_STS_TOKEN_ENV = "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_STS_TOKEN"

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
            project = self.sls_project
            if not project:
                return None
            logstore = self.sls_logstore or DEFAULT_SLS_LOGSTORE
            return f"sls://{project}/{logstore}"
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
        _logger.warning("Invalid multimodal upload_mode %r, fallback to none", value)
        return "none"
    return mode


def normalize_multimodal_hook_name(value: Any) -> Optional[str]:
    """Return normalized hook name, or None if unsupported."""
    if value is None:
        return None
    hook_name = str(value).strip().lower()
    if hook_name not in _VALID_HOOKS:
        _logger.warning("Unsupported multimodal hook name: %r", value)
        return None
    return hook_name


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
    uploader_hook_name = normalize_multimodal_hook_name(
        os.getenv(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER,
            DEFAULT_MULTIMODAL_UPLOADER_HOOK,
        )
    ) or DEFAULT_MULTIMODAL_UPLOADER_HOOK
    pre_uploader_hook_name = normalize_multimodal_hook_name(
        os.getenv(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER,
            DEFAULT_MULTIMODAL_PRE_UPLOADER_HOOK,
        )
    ) or DEFAULT_MULTIMODAL_PRE_UPLOADER_HOOK

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
    }

    for key, value in fields.items():
        if value is None or key not in merged:
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


def update_multimodal_runtime_config(**fields: Any) -> MultimodalConfigSnapshot:
    return _runtime_config.update(**fields)
