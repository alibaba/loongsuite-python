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

import os
import tempfile
import unittest
from dataclasses import dataclass
from typing import Any, Callable, List, Optional
from unittest.mock import patch

from opentelemetry.trace import SpanContext
from opentelemetry.util.genai._multimodal_upload._base import (  # pylint: disable=no-name-in-module
    PreUploader,
    PreUploadItem,
    Uploader,
    UploadItem,
)
from opentelemetry.util.genai._multimodal_upload.config import (  # pylint: disable=no-name-in-module
    DEFAULT_SLS_LOGSTORE,
    UPLOADER_GENERATION_FIELDS,
    get_multimodal_config_snapshot,
    normalize_multimodal_hook_name,
    update_multimodal_runtime_config,
)
from opentelemetry.util.genai.extended_environment_variables import (  # pylint: disable=no-name-in-module
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER,
)

from .multimodal_test_helpers import (
    get_default_pre_uploader_hook_name,
    get_default_uploader_hook_name,
    reload_multimodal_upload_hook_module,
    reset_multimodal_runtime_state_for_test,
)


@dataclass
class FakeEntryPoint:
    name: str
    load: Callable[[], Callable[..., Any]]


class TestMultimodalRuntimeConfig(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_env_fallback_defaults(self) -> None:
        reset_multimodal_runtime_state_for_test()
        snapshot = get_multimodal_config_snapshot()
        self.assertEqual(snapshot.upload_mode, "none")
        self.assertFalse(snapshot.download_enabled)
        self.assertEqual(
            snapshot.uploader_hook_name, get_default_uploader_hook_name()
        )
        self.assertEqual(
            snapshot.pre_uploader_hook_name, get_default_pre_uploader_hook_name()
        )
        self.assertIsNone(snapshot.effective_storage_base_path)
        self.assertFalse(snapshot.process_input)
        self.assertFalse(snapshot.process_output)

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED: "true",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "arms",
        },
        clear=True,
    )
    def test_env_arms_hook_builds_sls_effective_path(self) -> None:
        reset_multimodal_runtime_state_for_test()
        snapshot = get_multimodal_config_snapshot()
        self.assertEqual(snapshot.uploader_hook_name, "arms")
        self.assertTrue(snapshot.process_input)
        self.assertTrue(snapshot.process_output)
        self.assertTrue(snapshot.download_enabled)
        self.assertIsNone(snapshot.effective_storage_base_path)

    def test_upload_mode_input_and_output_flags(self) -> None:
        input_only = update_multimodal_runtime_config(upload_mode="input")
        self.assertTrue(input_only.process_input)
        self.assertFalse(input_only.process_output)

        output_only = update_multimodal_runtime_config(upload_mode="output")
        self.assertFalse(output_only.process_input)
        self.assertTrue(output_only.process_output)

    def test_arms_default_logstore_when_project_only(self) -> None:
        after = update_multimodal_runtime_config(
            uploader_hook_name="arms",
            sls_project="project-a",
        )
        self.assertIsNone(after.sls_logstore)
        self.assertEqual(
            after.effective_storage_base_path,
            f"sls://project-a/{DEFAULT_SLS_LOGSTORE}",
        )

    def test_strategy_update_does_not_bump_uploader_generation(self) -> None:
        before = get_multimodal_config_snapshot()
        after = update_multimodal_runtime_config(
            upload_mode="input",
            download_enabled=True,
            local_file_enabled="",
        )
        self.assertGreater(after.version, before.version)
        self.assertGreater(after.strategy_version, before.strategy_version)
        self.assertEqual(after.uploader_generation, before.uploader_generation)
        self.assertFalse(after.local_file_enabled)

    def test_uploader_generation_fields_bump_generation(self) -> None:
        before = get_multimodal_config_snapshot()
        after = update_multimodal_runtime_config(
            storage_base_path="oss://bucket/prefix",
            uploader_hook_name="fs",
            pre_uploader_hook_name="fs",
        )
        self.assertGreater(after.uploader_generation, before.uploader_generation)
        self.assertEqual(after.storage_base_path, "oss://bucket/prefix")
        self.assertEqual(after.effective_storage_base_path, "oss://bucket/prefix")

    def test_sls_project_change_bumps_uploader_generation(self) -> None:
        before = get_multimodal_config_snapshot()
        after = update_multimodal_runtime_config(
            uploader_hook_name="arms",
            pre_uploader_hook_name="arms",
            sls_project="project-a",
            sls_logstore="logstore-a",
        )
        self.assertGreater(after.uploader_generation, before.uploader_generation)
        self.assertEqual(after.uploader_hook_name, "arms")
        self.assertEqual(
            after.effective_storage_base_path,
            "sls://project-a/logstore-a",
        )

    def test_logstore_change_bumps_uploader_generation(self) -> None:
        update_multimodal_runtime_config(
            uploader_hook_name="arms",
            pre_uploader_hook_name="arms",
            sls_project="p",
            sls_logstore="l1",
        )
        before = get_multimodal_config_snapshot()
        after = update_multimodal_runtime_config(sls_logstore="l2")
        self.assertGreater(after.uploader_generation, before.uploader_generation)
        self.assertIn("sls_logstore", UPLOADER_GENERATION_FIELDS)
        self.assertEqual(after.effective_storage_base_path, "sls://p/l2")

    def test_invalid_hook_name_is_rejected(self) -> None:
        before = get_multimodal_config_snapshot()
        after = update_multimodal_runtime_config(uploader_hook_name="invalid")
        self.assertEqual(after, before)

    def test_invalid_pre_uploader_hook_is_rejected(self) -> None:
        before = get_multimodal_config_snapshot()
        after = update_multimodal_runtime_config(
            pre_uploader_hook_name="invalid"
        )
        self.assertEqual(after, before)

    def test_invalid_upload_mode_fallback_to_none(self) -> None:
        after = update_multimodal_runtime_config(upload_mode="invalid-mode")
        self.assertEqual(after.upload_mode, "none")

    def test_empty_upload_mode_fallback_to_none(self) -> None:
        after = update_multimodal_runtime_config(upload_mode="")
        self.assertEqual(after.upload_mode, "none")

    def test_no_op_update_keeps_snapshot(self) -> None:
        before = get_multimodal_config_snapshot()
        after = update_multimodal_runtime_config(upload_mode=before.upload_mode)
        self.assertEqual(after, before)

    def test_blank_sls_project_is_coalesced_to_none(self) -> None:
        update_multimodal_runtime_config(
            uploader_hook_name="arms",
            sls_project="project-a",
        )
        after = update_multimodal_runtime_config(sls_project="   ")
        self.assertIsNone(after.sls_project)

    def test_allowed_root_paths_normalized_to_absolute_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            after = update_multimodal_runtime_config(
                allowed_root_paths=f"{tmpdir}, {tmpdir}"
            )
            self.assertEqual(len(after.allowed_root_paths), 1)
            self.assertTrue(os.path.isabs(after.allowed_root_paths[0]))

    def test_allowed_root_paths_accepts_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            after = update_multimodal_runtime_config(
                allowed_root_paths=[tmpdir, tmpdir]
            )
            self.assertEqual(len(after.allowed_root_paths), 1)

    def test_normalize_multimodal_hook_name_none(self) -> None:
        self.assertIsNone(normalize_multimodal_hook_name(None))

    @patch.dict("os.environ", {}, clear=True)
    def test_none_to_both_can_reload_uploader_pair(self) -> None:
        module = reload_multimodal_upload_hook_module()

        uploader, pre_uploader = module.get_or_rebuild_uploader_pair()
        self.assertIsNone(uploader)
        self.assertIsNone(pre_uploader)

        with patch.dict(
            "os.environ",
            {
                OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
                OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH: "file:///tmp/mm",
                OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "fs",
                OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: "fs",
            },
            clear=False,
        ):
            module = reload_multimodal_upload_hook_module()

            class FakeUploader(Uploader):
                def upload(
                    self,
                    item: UploadItem,
                    *,
                    skip_if_exists: bool = True,
                ) -> bool:
                    return True

                def shutdown(self, timeout: float = 10.0) -> None:
                    return None

            class FakePreUploader(PreUploader):
                def pre_upload(
                    self,
                    span_context: Optional[SpanContext],
                    start_time_utc_nano: int,
                    input_messages: Optional[List[Any]],
                    output_messages: Optional[List[Any]],
                    config_snapshot: Optional[Any] = None,
                ) -> List[PreUploadItem]:
                    return []

            def fake_entry_points(group: str) -> List[FakeEntryPoint]:
                def uploader_loader() -> Callable[..., FakeUploader]:
                    def build(_snapshot: Any = None) -> FakeUploader:
                        return FakeUploader()

                    return build

                def pre_uploader_loader() -> Callable[..., FakePreUploader]:
                    def build(_snapshot: Any = None) -> FakePreUploader:
                        return FakePreUploader()

                    return build

                if group == "opentelemetry_genai_multimodal_uploader":
                    return [FakeEntryPoint("fs", uploader_loader)]
                if group == "opentelemetry_genai_multimodal_pre_uploader":
                    return [FakeEntryPoint("fs", pre_uploader_loader)]
                return []

            with patch.object(
                module, "_iter_entry_points", side_effect=fake_entry_points
            ):
                uploader, pre_uploader = module.get_or_rebuild_uploader_pair()

        self.assertIsNotNone(uploader)
        self.assertIsNotNone(pre_uploader)


if __name__ == "__main__":
    unittest.main()
