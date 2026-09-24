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

# Aliyun Python Agent Extension
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional
from unittest import TestCase
from unittest.mock import patch

from opentelemetry.util.genai._multimodal_upload._base import (
    PreUploader,
    PreUploadItem,
    Uploader,
    UploadItem,
)
from opentelemetry.util.genai._multimodal_upload.config import (  # pylint: disable=no-name-in-module
    MultimodalConfigSnapshot,
    update_multimodal_runtime_config,
)
from opentelemetry.util.genai.extended_environment_variables import (  # pylint: disable=no-name-in-module
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER,
)

from .multimodal_test_helpers import (
    get_default_pre_uploader_hook_name,
    get_default_uploader_hook_name,
    reload_multimodal_upload_hook_module,
)


class FakeUploader(Uploader):
    def upload(self, item: UploadItem, *, skip_if_exists: bool = True) -> bool:
        return True

    def shutdown(self, timeout: float = 10.0) -> None:
        return None


class FakePreUploader(PreUploader):
    def pre_upload(
        self,
        span_context: Optional[Any],
        start_time_utc_nano: int,
        input_messages: Optional[list[Any]],
        output_messages: Optional[list[Any]],
        config_snapshot: Optional[Any] = None,
    ) -> list[PreUploadItem]:
        return []


class InvalidHookResult:
    pass


@dataclass
class FakeEntryPoint:
    name: str
    load: Callable[[], Callable[[], Any]]


class TestMultimodalUploadHook(TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_get_or_load_without_uploader_env(self):
        module = reload_multimodal_upload_hook_module()
        uploader, pre_uploader = module.get_or_load_uploader_pair()
        self.assertIsNone(uploader)
        self.assertIsNone(pre_uploader)

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: "fs",
        },
        clear=True,
    )
    def test_getters_do_not_trigger_loading(self):
        module = reload_multimodal_upload_hook_module()
        with patch.object(module, "_iter_entry_points") as mock_iter:
            self.assertIsNone(module.get_uploader())
            self.assertIsNone(module.get_pre_uploader())
            self.assertEqual(module.get_uploader_pair(), (None, None))
        mock_iter.assert_not_called()

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: "fs",
        },
        clear=True,
    )
    def test_load_hooks_success(self):
        module = reload_multimodal_upload_hook_module()
        calls = {"uploader": 0, "pre": 0}

        def uploader_hook():
            calls["uploader"] += 1
            return FakeUploader()

        def pre_hook():
            calls["pre"] += 1
            return FakePreUploader()

        def fake_entry_points(group: str) -> list[FakeEntryPoint]:
            if group == "opentelemetry_genai_multimodal_uploader":
                return [FakeEntryPoint("fs", lambda: uploader_hook)]
            if group == "opentelemetry_genai_multimodal_pre_uploader":
                return [FakeEntryPoint("fs", lambda: pre_hook)]
            return []

        with patch.object(
            module, "_iter_entry_points", side_effect=fake_entry_points
        ):
            uploader, pre_uploader = module.get_or_load_uploader_pair()
        self.assertIsInstance(uploader, FakeUploader)
        self.assertIsInstance(pre_uploader, FakePreUploader)

        uploader2, pre_uploader2 = module.get_or_load_uploader_pair()
        self.assertIs(uploader2, uploader)
        self.assertIs(pre_uploader2, pre_uploader)
        self.assertEqual(calls["uploader"], 1)
        self.assertEqual(calls["pre"], 1)
        self.assertIs(module.get_uploader(), uploader)
        self.assertIs(module.get_pre_uploader(), pre_uploader)
        self.assertEqual(module.get_uploader_pair(), (uploader, pre_uploader))

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
        },
        clear=True,
    )
    def test_load_uploader_and_pre_uploader_use_configured_defaults(self):
        module = reload_multimodal_upload_hook_module()
        default_uploader = get_default_uploader_hook_name()
        default_pre_uploader = get_default_pre_uploader_hook_name()

        def uploader_factory():
            return FakeUploader()

        def pre_uploader_factory():
            return FakePreUploader()

        def load_uploader_factory():
            return uploader_factory

        def load_pre_uploader_factory():
            return pre_uploader_factory

        def fake_entry_points(group: str) -> list[FakeEntryPoint]:
            if group == "opentelemetry_genai_multimodal_uploader":
                return [
                    FakeEntryPoint(default_uploader, load_uploader_factory)
                ]
            if group == "opentelemetry_genai_multimodal_pre_uploader":
                return [
                    FakeEntryPoint(
                        default_pre_uploader, load_pre_uploader_factory
                    )
                ]
            return []

        with patch.object(
            module, "_iter_entry_points", side_effect=fake_entry_points
        ):
            uploader, pre_uploader = module.get_or_load_uploader_pair()
        self.assertIsInstance(uploader, FakeUploader)
        self.assertIsInstance(pre_uploader, FakePreUploader)

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: "fs",
        },
        clear=True,
    )
    def test_invalid_hook_result_fallback(self):
        module = reload_multimodal_upload_hook_module()

        def invalid_factory():
            return InvalidHookResult()

        def pre_uploader_factory():
            return FakePreUploader()

        def load_invalid_factory():
            return invalid_factory

        def load_pre_uploader_factory():
            return pre_uploader_factory

        def fake_entry_points(group: str) -> list[FakeEntryPoint]:
            if group == "opentelemetry_genai_multimodal_uploader":
                return [FakeEntryPoint("fs", load_invalid_factory)]
            if group == "opentelemetry_genai_multimodal_pre_uploader":
                return [FakeEntryPoint("fs", load_pre_uploader_factory)]
            return []

        with patch.object(
            module, "_iter_entry_points", side_effect=fake_entry_points
        ):
            uploader, pre_uploader = module.get_or_load_uploader_pair()
        self.assertIsNone(uploader)
        self.assertIsNone(pre_uploader)

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "none",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: "fs",
        },
        clear=True,
    )
    def test_upload_mode_none_disables_hooks(self):
        module = reload_multimodal_upload_hook_module()

        with patch.object(module, "_iter_entry_points") as mock_iter:
            uploader, pre_uploader = module.get_or_load_uploader_pair()
        self.assertIsNone(uploader)
        self.assertIsNone(pre_uploader)
        mock_iter.assert_not_called()


def _fake_fs_entry_points(
    uploader_factory: Callable[[], Uploader],
    pre_uploader_factory: Callable[[], PreUploader],
) -> Callable[[str], list[FakeEntryPoint]]:
    def fake_entry_points(group: str) -> list[FakeEntryPoint]:
        if group == "opentelemetry_genai_multimodal_uploader":
            return [FakeEntryPoint("fs", lambda: uploader_factory)]
        if group == "opentelemetry_genai_multimodal_pre_uploader":
            return [FakeEntryPoint("fs", lambda: pre_uploader_factory)]
        return []

    return fake_entry_points


class TestUploaderPairHotReload(TestCase):
    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH: "file:///tmp/mm",
        },
        clear=True,
    )
    def test_generation_bump_rebuilds_uploader_pair(self):
        module = reload_multimodal_upload_hook_module()
        created_uploaders: list[FakeUploader] = []

        def uploader_factory():
            uploader = FakeUploader()
            created_uploaders.append(uploader)
            return uploader

        def pre_uploader_factory() -> FakePreUploader:
            return FakePreUploader()

        with patch.object(
            module,
            "_iter_entry_points",
            side_effect=_fake_fs_entry_points(
                uploader_factory, pre_uploader_factory
            ),
        ):
            first_uploader, _ = module.get_or_rebuild_uploader_pair()
            update_multimodal_runtime_config(
                storage_base_path="file:///tmp/mm-v2"
            )
            second_uploader, _ = module.get_or_rebuild_uploader_pair()

        self.assertIsNotNone(first_uploader)
        self.assertIsNotNone(second_uploader)
        self.assertIsNot(first_uploader, second_uploader)
        self.assertEqual(len(created_uploaders), 2)

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH: "file:///tmp/mm",
        },
        clear=True,
    )
    def test_building_same_generation_returns_none_until_complete(self):
        module = reload_multimodal_upload_hook_module()
        load_started = threading.Event()
        release_load = threading.Event()

        def slow_load_pair(_snapshot: MultimodalConfigSnapshot):
            load_started.set()
            release_load.wait(timeout=5)
            return FakeUploader(), FakePreUploader()

        with patch.object(
            module, "_load_pair_from_snapshot", side_effect=slow_load_pair
        ):
            builder = threading.Thread(
                target=module.get_or_rebuild_uploader_pair,
                daemon=True,
            )
            builder.start()
            self.assertTrue(load_started.wait(timeout=5))

            concurrent_uploader, concurrent_pre = (
                module.get_or_rebuild_uploader_pair()
            )
            self.assertIsNone(concurrent_uploader)
            self.assertIsNone(concurrent_pre)

            release_load.set()
            builder.join(timeout=5)

            cached_uploader, cached_pre = module.get_or_rebuild_uploader_pair()
            self.assertIsInstance(cached_uploader, FakeUploader)
            self.assertIsInstance(cached_pre, FakePreUploader)

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH: "file:///tmp/mm",
        },
        clear=True,
    )
    def test_stale_build_discarded_when_generation_advances_during_load(self):
        module = reload_multimodal_upload_hook_module()
        shutdown_calls = {"count": 0}
        original_schedule = module._schedule_retired_pair_shutdown

        def track_shutdown(
            pair: tuple[Optional[Uploader], Optional[PreUploader]],
        ) -> None:
            shutdown_calls["count"] += 1
            original_schedule(pair)

        def load_pair_and_bump_generation(
            _snapshot: MultimodalConfigSnapshot,
        ):
            update_multimodal_runtime_config(
                storage_base_path="file:///tmp/mm-new"
            )
            return FakeUploader(), FakePreUploader()

        with patch.object(
            module,
            "_load_pair_from_snapshot",
            side_effect=load_pair_and_bump_generation,
        ):
            with patch.object(
                module,
                "_schedule_retired_pair_shutdown",
                side_effect=track_shutdown,
            ):
                uploader, pre_uploader = module.get_or_rebuild_uploader_pair()

        self.assertIsNone(uploader)
        self.assertIsNone(pre_uploader)
        self.assertEqual(shutdown_calls["count"], 1)
        self.assertIsNone(module.get_uploader())
        self.assertIsNone(module.get_pre_uploader())

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: "fs",
        },
        clear=True,
    )
    def test_failed_generation_is_not_retried(self):
        module = reload_multimodal_upload_hook_module()
        load_calls = {"count": 0}

        def counting_load_pair(_snapshot: MultimodalConfigSnapshot):
            load_calls["count"] += 1
            return None, None

        with patch.object(
            module, "_load_pair_from_snapshot", side_effect=counting_load_pair
        ):
            first = module.get_or_rebuild_uploader_pair()
            second = module.get_or_rebuild_uploader_pair()

        self.assertEqual(first, (None, None))
        self.assertEqual(second, (None, None))
        self.assertEqual(load_calls["count"], 1)

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH: "file:///tmp/mm",
        },
        clear=True,
    )
    def test_same_generation_keeps_cached_pair(self):
        module = reload_multimodal_upload_hook_module()
        load_calls = {"count": 0}

        def counting_load_pair(_snapshot: MultimodalConfigSnapshot):
            load_calls["count"] += 1
            return FakeUploader(), FakePreUploader()

        with patch.object(
            module, "_load_pair_from_snapshot", side_effect=counting_load_pair
        ):
            first = module.get_or_rebuild_uploader_pair()
            second = module.get_or_rebuild_uploader_pair()

        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        self.assertEqual(load_calls["count"], 1)
