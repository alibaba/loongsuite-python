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

"""Test helpers for resetting multimodal runtime singleton state."""

from __future__ import annotations

import importlib
from types import ModuleType

from opentelemetry.util.genai._multimodal_upload import (  # pylint: disable=no-name-in-module
    config as multimodal_config,
)
from opentelemetry.util.genai._multimodal_upload import (  # pylint: disable=no-name-in-module
    multimodal_upload_hook,
)

MULTIMODAL_UPLOAD_HOOK_MODULE = (
    "opentelemetry.util.genai._multimodal_upload.multimodal_upload_hook"
)


def reset_multimodal_runtime_config_for_test():
    """Reset process-wide multimodal config snapshot from current env."""
    multimodal_config._runtime_config = (  # pylint: disable=protected-access
        multimodal_config.MultimodalRuntimeConfig()
    )
    return multimodal_config._runtime_config.get_snapshot()  # pylint: disable=protected-access


def reset_multimodal_runtime_state_for_test() -> None:
    """Reset multimodal config snapshot and cached uploader pair state."""
    reset_multimodal_runtime_config_for_test()
    hook = multimodal_upload_hook
    with hook._uploader_pair_lock:  # pylint: disable=protected-access
        hook._uploader = None  # pylint: disable=protected-access
        hook._pre_uploader = None  # pylint: disable=protected-access
        hook._uploader_generation = -1  # pylint: disable=protected-access
        hook._failed_generation = -1  # pylint: disable=protected-access
        hook._building_generation = -1  # pylint: disable=protected-access


def get_default_uploader_hook_name() -> str:
    """Return bootstrap default uploader hook (community ``fs``, enterprise ``arms``)."""
    return multimodal_config.DEFAULT_MULTIMODAL_UPLOADER_HOOK


def get_default_pre_uploader_hook_name() -> str:
    """Return bootstrap default pre-uploader hook (community ``fs``, enterprise ``arms``)."""
    return multimodal_config.DEFAULT_MULTIMODAL_PRE_UPLOADER_HOOK


def reload_multimodal_upload_hook_module() -> ModuleType:
    """Reset runtime state and reload the hook module under current env."""
    reset_multimodal_runtime_state_for_test()
    module = importlib.import_module(MULTIMODAL_UPLOAD_HOOK_MODULE)
    return importlib.reload(module)
