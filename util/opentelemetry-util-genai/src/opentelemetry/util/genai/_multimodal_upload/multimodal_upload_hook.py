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

import logging
import threading
from importlib import metadata
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from opentelemetry.util.genai._multimodal_upload.config import (  # pylint: disable=no-name-in-module
    MultimodalConfigSnapshot,
    get_multimodal_config_snapshot,
)

from ._base import PreUploader, Uploader

_logger = logging.getLogger(__name__)

_MULTIMODAL_UPLOADER_ENTRY_POINT_GROUP = (
    "opentelemetry_genai_multimodal_uploader"
)
_MULTIMODAL_PRE_UPLOADER_ENTRY_POINT_GROUP = (
    "opentelemetry_genai_multimodal_pre_uploader"
)

_UPLOAD_MODE_NONE = "none"

_uploader: Optional[Uploader] = None
_pre_uploader: Optional[PreUploader] = None
_uploader_generation = -1
_failed_generation = -1
_building_generation = -1
_uploader_pair_lock = threading.RLock()


def _iter_entry_points(group: str) -> list[Any]:
    eps = metadata.entry_points()
    if hasattr(eps, "select"):
        return list(eps.select(group=group))  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType， reportAttributeAccessIssue]
    legacy_group_eps = eps[group] if group in eps else []
    return list(legacy_group_eps)


@runtime_checkable
class UploaderHook(Protocol):
    def __call__(
        self,
        snapshot: Optional[MultimodalConfigSnapshot] = None,
    ) -> Optional[Uploader]: ...


@runtime_checkable
class PreUploaderHook(Protocol):
    def __call__(
        self,
        snapshot: Optional[MultimodalConfigSnapshot] = None,
    ) -> Optional[PreUploader]: ...


def _call_hook(
    hook: Callable[..., Any],
    snapshot: MultimodalConfigSnapshot,
) -> Optional[object]:
    try:
        return hook(snapshot)
    except TypeError:
        return hook()


def _load_by_name(
    *,
    hook_name: str,
    group: str,
    snapshot: MultimodalConfigSnapshot,
) -> Optional[object]:
    for entry_point in _iter_entry_points(group):
        name = str(entry_point.name)
        if name != hook_name:
            continue
        try:
            hook = entry_point.load()
            return _call_hook(hook, snapshot)
        except Exception:  # pylint: disable=broad-except
            _logger.exception("%s hook %s configuration failed", group, name)
            return None
    return None


def _is_complete_pair(
    pair: tuple[Optional[Uploader], Optional[PreUploader]],
) -> bool:
    uploader, pre_uploader = pair
    return uploader is not None and pre_uploader is not None


def _schedule_retired_pair_shutdown(
    pair: tuple[Optional[Uploader], Optional[PreUploader]],
) -> None:
    uploader, pre_uploader = pair
    if uploader is None and pre_uploader is None:
        return

    def _shutdown() -> None:
        try:
            if pre_uploader is not None:
                pre_uploader.shutdown()
        except Exception:  # pylint: disable=broad-except
            _logger.debug("Failed to shutdown retired pre-uploader", exc_info=True)
        try:
            if uploader is not None:
                uploader.shutdown()
        except Exception:  # pylint: disable=broad-except
            _logger.debug("Failed to shutdown retired uploader", exc_info=True)

    thread = threading.Thread(
        target=_shutdown,
        name="multimodal-uploader-retire",
        daemon=True,
    )
    thread.start()


def load_uploader_hook(
    snapshot: Optional[MultimodalConfigSnapshot] = None,
) -> Optional[Uploader]:
    """Load multimodal uploader hook from entry points.

    Mechanism:
    - read hook name from runtime snapshot
      (`otel.instrumentation.genai.multimodal.uploader`, default: ``fs``)
    - resolve hook factory from entry-point group
      ``opentelemetry_genai_multimodal_uploader``
    - call hook factory with snapshot (legacy zero-arg hooks still supported)
    - validate returned object type (``Uploader``)
    """
    cfg = snapshot or get_multimodal_config_snapshot()
    if cfg.upload_mode == _UPLOAD_MODE_NONE:
        return None

    hook_name = cfg.uploader_hook_name or None
    if not hook_name:
        return None

    uploader = _load_by_name(
        hook_name=hook_name,
        group=_MULTIMODAL_UPLOADER_ENTRY_POINT_GROUP,
        snapshot=cfg,
    )
    if uploader is None:
        return None
    if not isinstance(uploader, Uploader):
        _logger.debug("%s is not a valid Uploader", hook_name)
        return None
    _logger.debug("Using multimodal uploader hook %s", hook_name)
    return uploader


def load_pre_uploader_hook(
    snapshot: Optional[MultimodalConfigSnapshot] = None,
) -> Optional[PreUploader]:
    """Load multimodal pre-uploader hook from entry points.

    Mechanism:
    - read hook name from runtime snapshot
      (`otel.instrumentation.genai.multimodal.pre-uploader`, default: ``fs``)
    - resolve hook factory from entry-point group
      ``opentelemetry_genai_multimodal_pre_uploader``
    - call hook factory with snapshot (legacy zero-arg hooks still supported)
    - validate returned object type (``PreUploader``)
    """
    cfg = snapshot or get_multimodal_config_snapshot()
    if cfg.upload_mode == _UPLOAD_MODE_NONE:
        return None

    hook_name = cfg.pre_uploader_hook_name or None
    if not hook_name:
        return None
    pre_uploader = _load_by_name(
        hook_name=hook_name,
        group=_MULTIMODAL_PRE_UPLOADER_ENTRY_POINT_GROUP,
        snapshot=cfg,
    )
    if pre_uploader is None:
        return None
    if not isinstance(pre_uploader, PreUploader):
        _logger.debug("%s is not a valid PreUploader", hook_name)
        return None
    _logger.debug("Using multimodal pre-uploader hook %s", hook_name)
    return pre_uploader


def _load_pair_from_snapshot(
    snapshot: MultimodalConfigSnapshot,
) -> tuple[Optional[Uploader], Optional[PreUploader]]:
    uploader = load_uploader_hook(snapshot)
    pre_uploader = load_pre_uploader_hook(snapshot)
    if uploader is None or pre_uploader is None:
        return None, None
    return uploader, pre_uploader


def get_or_rebuild_uploader_pair(
    snapshot: Optional[MultimodalConfigSnapshot] = None,
) -> tuple[Optional[Uploader], Optional[PreUploader]]:
    """Get or rebuild uploader/pre-uploader pair for the current generation.

    Generation-aware hot reload:
    - cache hit: return cached pair when ``uploader_generation`` unchanged
    - cache miss: load a new pair outside the lock, then install atomically
    - in-flight build: concurrent callers for the same generation get ``(None, None)``
    - stale build: if generation advanced during load, discard the new pair
    - failed generation: remember load failure and skip retry for that generation
    """
    global _uploader  # pylint: disable=global-statement
    global _pre_uploader  # pylint: disable=global-statement
    global _uploader_generation  # pylint: disable=global-statement
    global _failed_generation  # pylint: disable=global-statement
    global _building_generation  # pylint: disable=global-statement

    cfg = snapshot or get_multimodal_config_snapshot()
    if cfg.upload_mode == _UPLOAD_MODE_NONE:
        return None, None

    generation = cfg.uploader_generation

    with _uploader_pair_lock:
        if _failed_generation == generation:
            return None, None
        if (
            _uploader_generation == generation
            and _uploader is not None
            and _pre_uploader is not None
        ):
            return _uploader, _pre_uploader
        if _building_generation == generation:
            return None, None
        _building_generation = generation

    new_pair = _load_pair_from_snapshot(cfg)

    with _uploader_pair_lock:
        _building_generation = -1

        if get_multimodal_config_snapshot().uploader_generation != generation:
            _schedule_retired_pair_shutdown(new_pair)
            return None, None

        if not _is_complete_pair(new_pair):
            _failed_generation = generation
            return None, None

        old_pair = (_uploader, _pre_uploader)
        _uploader, _pre_uploader = new_pair
        _uploader_generation = generation
        _schedule_retired_pair_shutdown(old_pair)
        return _uploader, _pre_uploader


def get_or_load_uploader_pair(
    snapshot: Optional[MultimodalConfigSnapshot] = None,
) -> tuple[Optional[Uploader], Optional[PreUploader]]:
    """Backward-compatible alias for generation-aware pair loading."""
    return get_or_rebuild_uploader_pair(snapshot)


def get_uploader_pair() -> tuple[Optional[Uploader], Optional[PreUploader]]:
    """Return cached uploader pair without triggering lazy loading."""
    return _uploader, _pre_uploader


def get_or_load_uploader(
    snapshot: Optional[MultimodalConfigSnapshot] = None,
) -> Optional[Uploader]:
    """Get uploader and trigger lazy loading when needed."""
    return get_or_rebuild_uploader_pair(snapshot)[0]


def get_or_load_pre_uploader(
    snapshot: Optional[MultimodalConfigSnapshot] = None,
) -> Optional[PreUploader]:
    """Get pre-uploader and trigger lazy loading when needed."""
    return get_or_rebuild_uploader_pair(snapshot)[1]


def get_uploader() -> Optional[Uploader]:
    """Return cached uploader without triggering lazy loading."""
    return _uploader


def get_pre_uploader() -> Optional[PreUploader]:
    """Return cached pre-uploader without triggering lazy loading."""
    return _pre_uploader
