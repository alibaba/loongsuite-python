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
from unittest.mock import MagicMock, patch

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


class TestRetiredPairShutdownWorker(TestCase):
    def _reload_module(self):
        module = reload_multimodal_upload_hook_module()
        self.addCleanup(module._shutdown_retired_pair_worker_for_test)
        return module

    def test_empty_pair_does_not_start_worker(self):
        module = self._reload_module()

        module._schedule_retired_pair_shutdown((None, None))

        self.assertIsNone(module._retired_pair_worker)

    def test_reuses_single_worker_and_does_not_block_scheduler(self):
        module = self._reload_module()
        shutdown_started = threading.Event()
        release_shutdown = threading.Event()
        first_uploader = MagicMock(spec=Uploader)
        first_pre_uploader = MagicMock(spec=PreUploader)
        second_uploader = MagicMock(spec=Uploader)
        second_pre_uploader = MagicMock(spec=PreUploader)
        shutdown_order = []

        def block_first_shutdown():
            shutdown_order.append("first-pre")
            shutdown_started.set()
            release_shutdown.wait(timeout=5)

        first_uploader.shutdown.side_effect = lambda: shutdown_order.append(
            "first-uploader"
        )
        second_pre_uploader.shutdown.side_effect = lambda: (
            shutdown_order.append("second-pre")
        )
        second_uploader.shutdown.side_effect = lambda: shutdown_order.append(
            "second-uploader"
        )
        first_pre_uploader.shutdown.side_effect = block_first_shutdown
        module._schedule_retired_pair_shutdown(
            (first_uploader, first_pre_uploader)
        )
        self.assertTrue(shutdown_started.wait(timeout=5))
        worker = module._retired_pair_worker
        self.assertIsNotNone(worker)

        module._schedule_retired_pair_shutdown(
            (second_uploader, second_pre_uploader)
        )

        self.assertIs(module._retired_pair_worker, worker)
        assert worker is not None
        self.assertTrue(worker.is_alive())
        second_pre_uploader.shutdown.assert_not_called()

        release_shutdown.set()
        module._retired_pair_queue.join()

        first_pre_uploader.shutdown.assert_called_once_with()
        first_uploader.shutdown.assert_called_once_with()
        second_pre_uploader.shutdown.assert_called_once_with()
        second_uploader.shutdown.assert_called_once_with()
        self.assertEqual(
            shutdown_order,
            [
                "first-pre",
                "first-uploader",
                "second-pre",
                "second-uploader",
            ],
        )

    def test_shutdown_failure_does_not_stop_later_cleanup(self):
        module = self._reload_module()
        first_uploader = MagicMock(spec=Uploader)
        first_pre_uploader = MagicMock(spec=PreUploader)
        second_uploader = MagicMock(spec=Uploader)
        second_pre_uploader = MagicMock(spec=PreUploader)
        first_pre_uploader.shutdown.side_effect = RuntimeError("boom")
        first_uploader.shutdown.side_effect = RuntimeError("boom")

        module._schedule_retired_pair_shutdown(
            (first_uploader, first_pre_uploader)
        )
        module._schedule_retired_pair_shutdown(
            (second_uploader, second_pre_uploader)
        )
        module._retired_pair_queue.join()

        first_pre_uploader.shutdown.assert_called_once_with()
        first_uploader.shutdown.assert_called_once_with()
        second_pre_uploader.shutdown.assert_called_once_with()
        second_uploader.shutdown.assert_called_once_with()

    def test_after_fork_reset_discards_inherited_worker_state(self):
        module = self._reload_module()
        previous_queue = module._retired_pair_queue
        previous_lock = module._retired_pair_worker_lock
        module._retired_pair_worker = MagicMock(spec=threading.Thread)

        module._reset_retired_pair_worker_after_fork()

        self.assertIsNot(module._retired_pair_queue, previous_queue)
        self.assertIsNot(module._retired_pair_worker_lock, previous_lock)
        self.assertIsNone(module._retired_pair_worker)


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
    def test_generation_churn_keeps_one_retirement_worker(self):
        module = reload_multimodal_upload_hook_module()
        self.addCleanup(module._shutdown_retired_pair_worker_for_test)
        release_shutdown = threading.Event()
        self.addCleanup(release_shutdown.set)
        shutdown_started = threading.Event()
        built_pairs = []

        for index in range(21):
            uploader = MagicMock(spec=Uploader, name=f"uploader-{index}")
            pre_uploader = MagicMock(
                spec=PreUploader, name=f"pre-uploader-{index}"
            )
            built_pairs.append((uploader, pre_uploader))

        def block_first_retirement():
            shutdown_started.set()
            release_shutdown.wait(timeout=5)

        built_pairs[0][1].shutdown.side_effect = block_first_retirement

        with patch.object(
            module,
            "_load_pair_from_snapshot",
            side_effect=built_pairs,
        ):
            module.get_or_rebuild_uploader_pair()
            for generation in range(1, 21):
                update_multimodal_runtime_config(
                    storage_base_path=f"file:///tmp/mm-{generation}"
                )
                module.get_or_rebuild_uploader_pair()
                if generation == 1:
                    self.assertTrue(shutdown_started.wait(timeout=5))

        retirement_threads = [
            thread
            for thread in threading.enumerate()
            if thread.name == "multimodal-uploader-retire"
        ]
        self.assertEqual(retirement_threads, [module._retired_pair_worker])
        self.assertEqual(module._retired_pair_queue.qsize(), 19)

        release_shutdown.set()
        module._retired_pair_queue.join()

        for uploader, pre_uploader in built_pairs[:-1]:
            pre_uploader.shutdown.assert_called_once_with()
            uploader.shutdown.assert_called_once_with()
        built_pairs[-1][0].shutdown.assert_not_called()
        built_pairs[-1][1].shutdown.assert_not_called()

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
        self.assertIsNone(module._retired_pair_worker)
