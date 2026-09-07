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

"""Multimodal upload public exports."""

from __future__ import annotations

from opentelemetry.util.genai._multimodal_upload._base import (
    PreUploader,
    PreUploadItem,
    Uploader,
    UploadItem,
)
from opentelemetry.util.genai._multimodal_upload.config import (  # pylint: disable=no-name-in-module
    PRESIGN_HOOK_NAME,
    get_multimodal_config_snapshot,
    normalize_oss_bucket,
    update_multimodal_runtime_config,
)
from opentelemetry.util.genai._multimodal_upload.multimodal_upload_hook import (  # pylint: disable=no-name-in-module
    get_or_load_pre_uploader,
    get_or_load_uploader,
    get_or_load_uploader_pair,
    get_or_rebuild_uploader_pair,
    get_pre_uploader,
    get_uploader,
    get_uploader_pair,
    load_pre_uploader_hook,
    load_uploader_hook,
)
from opentelemetry.util.genai._multimodal_upload.usage_recorder import (
    MultimodalUsageRecorder,
    get_multimodal_usage_recorder,
    provider_label_from_protocol,
    set_multimodal_usage_recorder,
)

try:
    from opentelemetry.util.genai._multimodal_upload.fs_uploader import (
        FsUploader,
    )
except ImportError:
    FsUploader = None

try:
    from opentelemetry.util.genai._multimodal_upload.pre_uploader import (
        MultimodalPreUploader,
    )
except ImportError:
    MultimodalPreUploader = None

try:
    from opentelemetry.util.genai._multimodal_upload.presign_uploader import (
        PresignUploader,
    )
except ImportError:
    PresignUploader = None

try:
    from opentelemetry.util.genai._multimodal_upload.presign_client import (
        MultimodalPresignClient,
    )
except ImportError:
    MultimodalPresignClient = None

__all__ = [
    "UploadItem",
    "PreUploadItem",
    "Uploader",
    "PreUploader",
    "MultimodalUsageRecorder",
    "get_multimodal_usage_recorder",
    "set_multimodal_usage_recorder",
    "provider_label_from_protocol",
    "load_uploader_hook",
    "load_pre_uploader_hook",
    "get_uploader_pair",
    "get_or_load_uploader",
    "get_or_load_pre_uploader",
    "get_or_load_uploader_pair",
    "get_or_rebuild_uploader_pair",
    "get_multimodal_config_snapshot",
    "update_multimodal_runtime_config",
    "normalize_oss_bucket",
    "PRESIGN_HOOK_NAME",
    "get_uploader",
    "get_pre_uploader",
]

if FsUploader is not None:
    __all__.append("FsUploader")
if MultimodalPreUploader is not None:
    __all__.append("MultimodalPreUploader")
if PresignUploader is not None:
    __all__.append("PresignUploader")
if MultimodalPresignClient is not None:
    __all__.append("MultimodalPresignClient")
