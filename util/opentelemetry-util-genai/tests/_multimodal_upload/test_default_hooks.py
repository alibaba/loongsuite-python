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

import logging
from unittest import TestCase
from unittest.mock import patch

from opentelemetry.util.genai._multimodal_upload.fs_uploader import (  # pylint: disable=no-name-in-module
    FsUploader,
    fs_uploader_hook,
)
from opentelemetry.util.genai._multimodal_upload.pre_uploader import (  # pylint: disable=no-name-in-module
    MultimodalPreUploader,
    fs_pre_uploader_hook,
)
from .multimodal_test_helpers import reset_multimodal_runtime_state_for_test
from opentelemetry.util.genai.extended_environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH,
)


class TestDefaultHooks(TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_fs_uploader_hook_returns_none_without_base_path(self):
        with self.assertLogs(level=logging.WARNING) as logs:
            uploader = fs_uploader_hook()
        self.assertIsNone(uploader)
        self.assertTrue(
            any(
                "multimodal uploader disabled" in message
                for message in logs.output
            )
        )

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH: "file:///tmp"
        },
        clear=True,
    )
    def test_fs_uploader_hook_returns_uploader(self):
        reset_multimodal_runtime_state_for_test()
        uploader = fs_uploader_hook()
        self.assertIsInstance(uploader, FsUploader)
        assert uploader is not None
        uploader.shutdown(timeout=0.1)

    @patch.dict("os.environ", {}, clear=True)
    def test_fs_pre_uploader_hook_returns_none_without_base_path(self):
        with self.assertLogs(level=logging.WARNING) as logs:
            pre_uploader = fs_pre_uploader_hook()
        self.assertIsNone(pre_uploader)
        self.assertTrue(
            any(
                "multimodal pre-uploader disabled" in message
                for message in logs.output
            )
        )

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH: "file:///tmp"
        },
        clear=True,
    )
    def test_fs_pre_uploader_hook_returns_pre_uploader(self):
        reset_multimodal_runtime_state_for_test()
        pre_uploader = fs_pre_uploader_hook()
        self.assertIsInstance(pre_uploader, MultimodalPreUploader)
        assert pre_uploader is not None
        pre_uploader.shutdown(timeout=0.1)
