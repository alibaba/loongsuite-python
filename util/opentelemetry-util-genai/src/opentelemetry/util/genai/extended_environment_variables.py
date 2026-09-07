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

OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH = (
    "OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH

The maximum length of message content to capture. Content exceeding this length will be truncated.
Defaults to 8192.
"""

OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_CAPTURE_STRATEGY = (
    "OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_CAPTURE_STRATEGY"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_CAPTURE_STRATEGY

The strategy for capturing message content. Must be one of ``span-attributes`` or ``event``.
Defaults to ``span-attributes``.
"""

# ============================================================================
# Multimodal Upload Environment Variables
#
# Similar to OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH in _upload/completion_hook.py,
# multimodal upload also needs base path configuration and behavior control.
# ============================================================================

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH

Base path for multimodal storage. Must be configured to enable multimodal upload.
Example: ``sls://`` for SLS storage.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE

Upload mode for multimodal data. Must be one of ``none``, ``input``, ``output``, or ``both``.
Defaults to ``none``.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED

Whether to download from external URI and re-upload to storage. Set to ``true`` or ``false``.
Defaults to ``false``.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY

Whether to verify SSL certificates when downloading external URI references.
Set to ``true`` or ``false``. Defaults to ``true``.
Disabling SSL verification may expose to man-in-the-middle attacks.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_AUDIO_CONVERSION_ENABLED = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_AUDIO_CONVERSION_ENABLED"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_AUDIO_CONVERSION_ENABLED

Whether to enable audio transcoding in multimodal pre-processing
(currently PCM16/L16/PCM to WAV).
Set to ``true`` or ``false``. Defaults to ``false``.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_LOCAL_FILE_ENABLED = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_LOCAL_FILE_ENABLED"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_LOCAL_FILE_ENABLED

Whether to allow the multimodal pipeline to read and upload files directly
from the local file system (supports ``file://`` URIs, absolute paths, and
relative paths).
Set to ``true`` or ``false``. Defaults to ``false``.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_ALLOWED_ROOT_PATHS = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_ALLOWED_ROOT_PATHS"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_ALLOWED_ROOT_PATHS

List of allowed root paths for local file access (comma separated).
Only files within these paths will be allowed for upload.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER

Select multimodal uploader hook name from entry point group
``opentelemetry_genai_multimodal_uploader``.
Defaults to ``fs`` when unset.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER

Select multimodal pre-uploader hook name from entry point group
``opentelemetry_genai_multimodal_pre_uploader``.
Defaults to ``fs`` when unset.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_BUCKET = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_BUCKET"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_BUCKET

Deprecated and ignored by the ``presign`` uploader: the backing bucket is owned
and decided by the server, so it cannot be selected by the agent. Use
``OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_PATH_PREFIX`` to group an
application's objects under a common path instead.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_PATH_PREFIX = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_PATH_PREFIX"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_PATH_PREFIX

Common path prefix applied to every multimodal object uploaded by this
application, for example ``my-app`` or ``my-app/images``. Objects are then
addressed as ``sls://{project}/{logstore}/{prefix}/{date}/{md5}.{ext}``.
Optional; when unset objects are stored directly under the logstore.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_ENDPOINT = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_ENDPOINT"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_ENDPOINT

Override the endpoint used to request presigned multimodal upload URLs.
Requires a base URL including its HTTP or HTTPS scheme, without a trailing
slash (for example ``https://cn-hangzhou.log.aliyuncs.com``). The URL is used
unchanged before appending the presign API path. When unset, the endpoint is resolved from the ARMS/SLS OneEndpoint state, which is
only available when running under an ARMS agent.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_TIMEOUT = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_TIMEOUT"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_TIMEOUT

Timeout in seconds for presign requests and presigned uploads.
Defaults to ``30``.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_LICENSE_KEY = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_LICENSE_KEY"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_LICENSE_KEY

License key authenticating presign requests. Falls back to ``ARMS_LICENSE_KEY``
so applications already running under an ARMS agent need no extra
configuration. Required by the ``presign`` uploader: without it the whole
multimodal upload chain degrades to disabled.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_WORKSPACE = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_WORKSPACE"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_WORKSPACE

CMS workspace the multimodal objects belong to. Falls back to
``ARMS_WORKSPACE``. Optional; the server derives a default workspace from the
license key when omitted.
"""
