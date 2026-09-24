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

"""FsUploader 单元测试

此测试文件测试基于 fsspec 的 FsUploader，依赖 fsspec/ossfs。
"""

import json as _json
import os
import threading
import time
from typing import Dict
from unittest.mock import MagicMock, patch

import fsspec
import httpx
import pytest

from opentelemetry.util.genai._multimodal_upload import FsUploader, UploadItem

# 使用测试文件自身作为测试内容
_THIS_FILE = os.path.abspath(__file__)


def _read_test_content() -> bytes:
    """读取测试文件自身作为测试内容"""
    with open(_THIS_FILE, "rb") as file_obj:
        return file_obj.read()


def test_upload_local_binary():
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "upload"))
    os.makedirs(base_dir, exist_ok=True)
    base_path = base_dir

    content = _read_test_content()

    uploader = FsUploader(base_path=base_path, max_workers=2)
    try:
        dst = os.path.join(base_path, "sample.bin")
        if os.path.exists(dst):
            os.remove(dst)
        meta_path = dst + ".meta"
        if os.path.exists(meta_path):
            os.remove(meta_path)

        ok = uploader.upload(
            UploadItem(
                url="sample.bin",
                data=content,
                content_type="application/octet-stream",
                meta={"from": "test", "owner": "ci"},
            )
        )
        assert ok
        uploader.shutdown()

        assert os.path.exists(dst)
        assert os.path.getsize(dst) == len(content)

        assert os.path.exists(meta_path)

        with open(meta_path, "r", encoding="utf-8") as mf:
            meta_loaded = _json.load(mf)
        assert meta_loaded == {"from": "test", "owner": "ci"}
    finally:
        try:
            uploader.shutdown()
        except (OSError, RuntimeError):
            pass


def test_upload_oss_binary_env():
    region_id = os.getenv("ARMS_REGION_ID", "")
    endpoint = "https://oss-" + region_id + ".aliyuncs.com"
    key = os.getenv("APSARA_APM_COLLECTOR_MULTIMODAL_OSS_ACCESS_KEY_ID")
    secret = os.getenv("APSARA_APM_COLLECTOR_MULTIMODAL_OSS_ACCESS_KEY_SECRET")
    storage_base_path = os.getenv(
        "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH", ""
    )

    # 提前检查必要的环境变量
    if not (
        region_id
        and key
        and secret
        and storage_base_path
        and "://" in storage_base_path
    ):
        pytest.skip(
            "OSS credentials not set in environment; source export_env.sh to run this test."
        )

    pytest.importorskip("ossfs")

    bucket = storage_base_path.split("://", 1)[1].split("/", 1)[0]

    storage_options: Dict[str, str] = {
        "endpoint": endpoint,
        "key": key,
        "secret": secret,
    }
    token = os.getenv("APSARA_APM_COLLECTOR_MULTIMODAL_OSS_STS_TOKEN")
    if token:
        storage_options["token"] = token

    content = _read_test_content()

    uploader = FsUploader(
        base_path=f"oss://{bucket}",
        storage_options=storage_options,
        max_workers=2,
    )
    key_path = "workspace/sample.bin"
    try:
        fs, _ = fsspec.url_to_fs(f"oss://{bucket}", **storage_options)
        full_uri = f"oss://{bucket}/{key_path}"
        try:
            if fs.exists(full_uri):
                fs.rm(full_uri)
            if fs.exists(full_uri + ".meta"):
                fs.rm(full_uri + ".meta")
        except (OSError, IOError):
            pass

        ok = uploader.upload(
            UploadItem(
                url=key_path,
                data=content,
                content_type="application/octet-stream",
                meta={"from": "test"},
            )
        )
        assert ok
        uploader.shutdown()

        assert fs.exists(full_uri)
    finally:
        try:
            uploader.shutdown()
        except (OSError, RuntimeError):
            pass


def test_max_queue_size_limit():
    """测试队列大小限制"""
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "upload_queue_test"))
    os.makedirs(base_dir, exist_ok=True)

    # 创建一个队列容量为 3 的 uploader，worker 数为 1 以便控制消费速度
    uploader = FsUploader(
        base_path=base_dir,
        max_workers=1,
        max_queue_size=3,
    )

    # 用于阻塞上传的事件
    block_event = threading.Event()
    original_write = uploader._write_file_with_optional_headers

    def slow_write(*args, **kwargs):
        block_event.wait()  # 等待解除阻塞
        return original_write(*args, **kwargs)

    uploader._write_file_with_optional_headers = slow_write

    try:
        # 前 3 个应该成功入队
        assert (
            uploader.upload(
                UploadItem(
                    url="file1.txt",
                    data=b"content1",
                    content_type="text/plain",
                    meta={},
                )
            )
            is True
        )
        assert (
            uploader.upload(
                UploadItem(
                    url="file2.txt",
                    data=b"content2",
                    content_type="text/plain",
                    meta={},
                )
            )
            is True
        )
        assert (
            uploader.upload(
                UploadItem(
                    url="file3.txt",
                    data=b"content3",
                    content_type="text/plain",
                    meta={},
                )
            )
            is True
        )

        # 第 4 个应该因队列满而失败
        assert (
            uploader.upload(
                UploadItem(
                    url="file4.txt",
                    data=b"content4",
                    content_type="text/plain",
                    meta={},
                )
            )
            is False
        )

        # 解除阻塞，让任务完成
        block_event.set()
        time.sleep(0.5)

        # 现在应该可以入队了
        assert (
            uploader.upload(
                UploadItem(
                    url="file5.txt",
                    data=b"content5",
                    content_type="text/plain",
                    meta={},
                )
            )
            is True
        )
    finally:
        block_event.set()
        uploader.shutdown(timeout=5.0)


def test_max_queue_bytes_limit():
    """测试队列字节数限制"""
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "upload_bytes_test"))
    os.makedirs(base_dir, exist_ok=True)

    # 创建一个字节限制为 100 的 uploader
    uploader = FsUploader(
        base_path=base_dir,
        max_workers=1,
        max_queue_size=100,  # 大队列容量
        max_queue_bytes=100,  # 100 字节限制
    )

    # 用于阻塞上传的事件
    block_event = threading.Event()
    original_write = uploader._write_file_with_optional_headers

    def slow_write(*args, **kwargs):
        block_event.wait()
        return original_write(*args, **kwargs)

    uploader._write_file_with_optional_headers = slow_write

    try:
        # 50 字节的内容
        content_50 = b"x" * 50

        # 前 2 个应该成功入队 (50 + 50 = 100)
        assert (
            uploader.upload(
                UploadItem(
                    url="file1.txt",
                    data=content_50,
                    content_type="text/plain",
                    meta={},
                )
            )
            is True
        )
        assert (
            uploader.upload(
                UploadItem(
                    url="file2.txt",
                    data=content_50,
                    content_type="text/plain",
                    meta={},
                )
            )
            is True
        )

        # 第 3 个应该因字节数超限而失败
        assert (
            uploader.upload(
                UploadItem(
                    url="file3.txt",
                    data=content_50,
                    content_type="text/plain",
                    meta={},
                )
            )
            is False
        )

        # 小内容也不行，因为已经到达 100 字节限制
        assert (
            uploader.upload(
                UploadItem(
                    url="small.txt",
                    data=b"tiny",
                    content_type="text/plain",
                    meta={},
                )
            )
            is False
        )

        # 解除阻塞，让任务完成
        block_event.set()
        time.sleep(0.5)

        # 现在应该可以入队了
        assert (
            uploader.upload(
                UploadItem(
                    url="file4.txt",
                    data=content_50,
                    content_type="text/plain",
                    meta={},
                )
            )
            is True
        )
    finally:
        block_event.set()
        uploader.shutdown(timeout=5.0)


class TestDownloadAndUpload:
    """测试下载-上传功能（DOWNLOAD_AND_UPLOAD 类型任务）"""

    @staticmethod
    def test_upload_with_source_uri_requires_no_content():
        """测试 source_uri 参数：不需要提供 data"""
        base_dir = os.path.abspath(
            os.path.join(os.getcwd(), "upload_source_uri_test")
        )
        os.makedirs(base_dir, exist_ok=True)

        uploader = FsUploader(base_path=base_dir, max_workers=1)
        try:
            # 没有 data 也没有 source_uri，应该失败
            assert (
                uploader.upload(
                    UploadItem(
                        url="fail.txt", content_type="text/plain", meta={}
                    )
                )
                is False
            )

            # 有 data 应该成功
            assert (
                uploader.upload(
                    UploadItem(
                        url="success1.txt",
                        data=b"content",
                        content_type="text/plain",
                        meta={},
                    )
                )
                is True
            )

            # 有 source_uri 应该成功（入队），但实际下载可能失败
            assert (
                uploader.upload(
                    UploadItem(
                        url="success2.txt",
                        source_uri="https://httpbin.org/bytes/100",
                        expected_size=100,
                        content_type="application/octet-stream",
                        meta={},
                    )
                )
                is True
            )
        finally:
            uploader.shutdown(timeout=5.0)

    @staticmethod
    def test_download_content_with_mock():
        """测试 _download_content 方法（使用 mock）"""
        base_dir = os.path.abspath(
            os.path.join(os.getcwd(), "upload_download_test")
        )
        os.makedirs(base_dir, exist_ok=True)

        uploader = FsUploader(base_path=base_dir, max_workers=1)

        try:
            # Mock httpx.Client 返回成功响应
            test_content = b"downloaded content"
            mock_response = MagicMock()
            mock_response.status_code = 200  # 设置状态码
            mock_response.iter_bytes.return_value = [test_content]
            mock_response.raise_for_status = MagicMock()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)

            mock_client = MagicMock()
            mock_client.stream.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)

            with patch("httpx.Client", return_value=mock_client):
                result = uploader._download_content(
                    "https://example.com/test.bin", max_size=1024
                )
                assert result == test_content

            # 测试超过大小限制
            large_content = b"x" * 100
            mock_response.iter_bytes.return_value = [large_content]

            with patch("httpx.Client", return_value=mock_client):
                result = uploader._download_content(
                    "https://example.com/large.bin",
                    max_size=50,  # 小于 100
                )
                assert result is None
        finally:
            uploader.shutdown(timeout=1.0)

    @staticmethod
    def test_download_content_exception_handling():
        """测试 _download_content 异常处理"""
        base_dir = os.path.abspath(
            os.path.join(os.getcwd(), "upload_download_exc_test")
        )
        os.makedirs(base_dir, exist_ok=True)

        uploader = FsUploader(base_path=base_dir, max_workers=1)

        try:
            # Mock httpx.Client 抛出 httpx.HTTPError 异常（更具体的异常类型）
            with patch(
                "httpx.Client", side_effect=httpx.ConnectError("Network error")
            ):
                result = uploader._download_content(
                    "https://example.com/error.bin", max_size=1024
                )
                assert result is None
        finally:
            uploader.shutdown(timeout=1.0)

    @staticmethod
    def test_queue_bytes_with_expected_size():
        """测试使用 expected_size 进行队列字节管理"""
        base_dir = os.path.abspath(
            os.path.join(os.getcwd(), "upload_expected_size_test")
        )
        os.makedirs(base_dir, exist_ok=True)

        uploader = FsUploader(
            base_path=base_dir,
            max_workers=1,
            max_queue_size=100,
            max_queue_bytes=100,  # 100 字节限制
        )

        # 阻塞上传
        block_event = threading.Event()
        original_do_upload = uploader._do_upload

        def slow_do_upload(*args, **kwargs):
            block_event.wait()
            return original_do_upload(*args, **kwargs)

        uploader._do_upload = slow_do_upload

        try:
            # 使用 expected_size 入队（50 + 50 = 100）
            assert (
                uploader.upload(
                    UploadItem(
                        url="file1.txt",
                        source_uri="https://example.com/file1",
                        expected_size=50,
                        content_type="application/octet-stream",
                        meta={},
                    )
                )
                is True
            )
            assert (
                uploader.upload(
                    UploadItem(
                        url="file2.txt",
                        source_uri="https://example.com/file2",
                        expected_size=50,
                        content_type="application/octet-stream",
                        meta={},
                    )
                )
                is True
            )

            # 第 3 个应该因字节数超限而失败
            assert (
                uploader.upload(
                    UploadItem(
                        url="file3.txt",
                        source_uri="https://example.com/file3",
                        expected_size=10,
                        content_type="application/octet-stream",
                        meta={},
                    )
                )
                is False
            )

        finally:
            block_event.set()
            uploader.shutdown(timeout=5.0)
