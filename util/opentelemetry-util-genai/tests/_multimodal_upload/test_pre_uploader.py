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

"""
Test general functionality of MultimodalPreUploader
Includes extension mapping, URL generation, meta processing, message handling, async metadata fetching, etc.
"""
# pylint: disable=too-many-lines

import asyncio
import base64
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
import respx

from opentelemetry.util.genai._multimodal_processing import (
    MultimodalProcessingMixin,
)
from opentelemetry.util.genai._multimodal_upload.pre_uploader import (
    _MAX_MULTIMODAL_DATA_SIZE,
    _MAX_MULTIMODAL_PARTS,
    MultimodalPreUploader,
    UriMetadata,
)
from opentelemetry.util.genai.types import Base64Blob, Blob, InputMessage, Uri

from .multimodal_test_helpers import reset_multimodal_runtime_state_for_test

# Test audio file directory for integration tests
TEST_AUDIO_DIR = Path(__file__).parent / "test_audio_samples"


@pytest.fixture(autouse=True)
def _default_upload_mode_enabled_for_tests():
    with patch.dict(
        "os.environ",
        {
            "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE": "both",
            "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED": "true",
        },
    ):
        reset_multimodal_runtime_state_for_test()
        yield


class TestPreUploadGeneral:
    """Test general functionality of MultimodalPreUploader"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        """Create PreUploader instance"""
        base_path = "/tmp/test_upload"
        extra_meta = {
            "workspaceId": "test_workspace",
            "serviceId": "test_service",
        }
        return MultimodalPreUploader(
            base_path=base_path, extra_meta=extra_meta
        )

    # ========== Extension Mapping Tests ==========

    @staticmethod
    @pytest.mark.parametrize(
        "mime_type,expected_ext",
        [
            # Audio formats
            ("audio/wav", "wav"),
            ("audio/mp3", "mp3"),
            ("audio/mpeg", "mp3"),  # Special mapping
            ("audio/aac", "aac"),
            ("audio/m4a", "m4a"),
            ("audio/amr", "amr"),
            ("audio/amr-wb", "amr"),  # Special mapping
            ("audio/3gpp", "3gp"),  # Special mapping
            ("audio/3gpp2", "3g2"),  # Special mapping
            ("audio/ogg", "ogg"),
            ("audio/flac", "flac"),
            ("audio/webm", "webm"),
            # Image formats
            ("image/jpeg", "jpg"),  # Special mapping
            ("image/png", "png"),
            # Edge cases
            ("audio/*", "bin"),  # Wildcard
            ("audio/unknown", "bin"),  # Unknown format
            ("video", "bin"),  # No slash
        ],
    )
    def test_extension_mapping(pre_uploader, mime_type, expected_ext):
        """Test extension mapping correctness"""
        ext = pre_uploader._ext_from_content_type(mime_type)
        assert ext == expected_ext, (
            f"MIME type {mime_type} extension mapping error, expected {expected_ext}, got {ext}"
        )

    # ========== URL Generation Tests ==========

    @staticmethod
    @pytest.mark.parametrize(
        "base_path",
        [
            "/tmp/test_upload",  # Without trailing slash
            "/tmp/test_upload/",  # With trailing slash
        ],
    )
    def test_url_generation_with_different_base_paths(base_path):
        """Test URL generation with/without trailing slash in base_path"""
        pre_uploader = MultimodalPreUploader(base_path=base_path)

        test_data = b"test data"

        part = Blob(content=test_data, mime_type="image/png", modality="image")

        input_message = InputMessage(role="user", parts=[part])
        input_messages = [input_message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 1
        # URL should correctly handle base_path trailing slash
        assert "/tmp/test_upload/" in uploads[0].url
        assert uploads[0].url.endswith(".png")

    # ========== Message Processing and Meta Fields Integration Tests ==========

    # pylint: disable=R0914
    @staticmethod
    def test_process_messages_and_meta(pre_uploader):
        """Test message processing and meta fields (integration test)"""
        test_data = b"test data"

        # Input message
        input_part = Blob(
            content=test_data, mime_type="image/png", modality="image"
        )
        input_message = InputMessage(role="user", parts=[input_part])
        input_messages = [input_message]

        # Output message
        output_part = Blob(
            content=test_data, mime_type="image/jpeg", modality="image"
        )
        output_message = InputMessage(role="assistant", parts=[output_part])
        output_messages = [output_message]

        # Test with span_context
        mock_span = Mock()
        mock_span.trace_id = 123456789
        mock_span.span_id = 987654321

        uploads = pre_uploader.pre_upload(
            span_context=mock_span,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=output_messages,
        )

        # Verify both input and output are processed
        assert len(uploads) == 2

        # Verify meta contains all necessary fields (with span_context)
        for upload in uploads:
            assert "workspaceId" in upload.meta
            assert "serviceId" in upload.meta
            assert "timestamp" in upload.meta
            assert "traceId" in upload.meta
            assert "spanId" in upload.meta

        # Test without span_context
        input_part2 = Blob(
            content=test_data, mime_type="image/png", modality="image"
        )
        input_message2 = InputMessage(role="user", parts=[input_part2])
        input_messages2 = [input_message2]

        uploads2 = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages2,
            output_messages=None,
        )

        assert len(uploads2) == 1
        # Verify meta doesn't contain trace/span fields
        meta = uploads2[0].meta
        assert "workspaceId" in meta
        assert "serviceId" in meta
        assert "timestamp" in meta
        assert "traceId" not in meta
        assert "spanId" not in meta

    @staticmethod
    def test_process_empty_messages(pre_uploader):
        """Test empty messages"""
        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=None,
            output_messages=None,
        )

        assert len(uploads) == 0

    # ========== Exception Handling Tests ==========

    # ========== Complete Workflow Tests ==========

    @staticmethod
    def test_audio_unknown_to_detected_workflow(pre_uploader):
        """Test audio/unknown auto-detection and conversion to correct format workflow"""
        filename = "test.wav"
        expected_mime = "audio/wav"
        expected_ext = "wav"

        filepath = TEST_AUDIO_DIR / filename
        if not filepath.exists():
            pytest.skip(f"Test audio file does not exist: {filepath}")

        # Read real audio file
        with open(filepath, "rb") as file_obj:
            audio_data = file_obj.read()

        # Create message with audio/unknown
        part = Blob(
            content=audio_data, mime_type="audio/unknown", modality="audio"
        )

        input_message = InputMessage(role="user", parts=[part])
        input_messages = [input_message]

        # Execute pre_upload workflow
        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        # Verify auto-detection and conversion successful
        assert len(uploads) == 1
        assert uploads[0].content_type == expected_mime, (
            f"MIME type should convert from audio/unknown to {expected_mime}"
        )
        assert uploads[0].url.endswith(f".{expected_ext}")


class TestPreUploadKeyGeneration:
    """Test remote URI key generation strategy"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        return MultimodalPreUploader(base_path="/tmp/test_upload")

    @staticmethod
    @pytest.mark.parametrize(
        "uri,etag,last_modified,expected_contains_url",
        [
            # Different ETags should generate different keys
            ("https://example.com/image.png", '"abc123"', None, True),
            ("https://example.com/image.png", '"def456"', None, True),
            # Same params should generate same key
            (
                "https://example.com/image.png?v=1",
                '"abc123"',
                "Wed, 01 Jan 2020 00:00:00 GMT",
                True,
            ),
            (
                "https://example.com/image.png?v=2",
                '"abc123"',
                "Wed, 01 Jan 2020 00:00:00 GMT",
                True,
            ),
        ],
    )
    def test_generate_remote_key_consistency(
        pre_uploader, uri, etag, last_modified, expected_contains_url
    ):
        """Test _generate_remote_key consistency"""
        key = pre_uploader._generate_remote_key(
            uri=uri,
            content_type="image/png",
            content_length=1024,
            etag=etag,
            last_modified=last_modified,
        )

        # key should be 32-char MD5
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    @staticmethod
    def test_generate_remote_key_same_etag_different_url_same_key(
        pre_uploader,
    ):
        """Test same key generation when URL query params differ but other metadata is same"""
        key1 = pre_uploader._generate_remote_key(
            uri="https://cdn.example.com/image.png?token=abc",
            content_type="image/png",
            content_length=1024,
            etag='"same-etag"',
            last_modified="Wed, 01 Jan 2020 00:00:00 GMT",
        )
        key2 = pre_uploader._generate_remote_key(
            uri="https://cdn.example.com/image.png?token=xyz",
            content_type="image/png",
            content_length=1024,
            etag='"same-etag"',
            last_modified="Wed, 01 Jan 2020 00:00:00 GMT",
        )

        # Different query params but same others should generate same key
        assert key1 == key2

    @staticmethod
    def test_generate_remote_key_different_domain_different_key(pre_uploader):
        """Test different keys for different domains (prevent CDN collisions)"""
        key1 = pre_uploader._generate_remote_key(
            uri="https://cdn1.example.com/image.png",
            content_type="image/png",
            content_length=1024,
            etag='"same-etag"',
            last_modified="Wed, 01 Jan 2020 00:00:00 GMT",
        )
        key2 = pre_uploader._generate_remote_key(
            uri="https://cdn2.example.com/image.png",
            content_type="image/png",
            content_length=1024,
            etag='"same-etag"',
            last_modified="Wed, 01 Jan 2020 00:00:00 GMT",
        )

        # Different domains should generate different keys
        assert key1 != key2


class TestPreUploadUri:
    """Test MultimodalPreUploader Uri handling"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        """Create PreUploader instance"""
        return MultimodalPreUploader(
            base_path="/tmp/test_upload",
            extra_meta={
                "workspaceId": "test_workspace",
                "serviceId": "test_service",
            },
        )

    # ========== Async Metadata Fetching Tests ==========

    @staticmethod
    @pytest.mark.asyncio
    async def test_fetch_one_metadata_async_success(pre_uploader):
        """Test successful async fetch of single URI metadata"""
        mock_response = MagicMock()
        mock_response.headers = {
            "Content-Type": "image/png",
            "Content-Range": "bytes 0-0/1024",
            "ETag": '"abc123"',
            "Last-Modified": "Wed, 01 Jan 2020 00:00:00 GMT",
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        uri, metadata = await pre_uploader._fetch_one_metadata_async(
            mock_client, "https://example.com/image.png"
        )

        assert uri == "https://example.com/image.png"
        assert metadata is not None
        assert metadata.content_type == "image/png"
        assert metadata.content_length == 1024
        assert metadata.etag == '"abc123"'
        assert metadata.last_modified == "Wed, 01 Jan 2020 00:00:00 GMT"

    @staticmethod
    @pytest.mark.asyncio
    async def test_fetch_one_metadata_async_no_content_type(pre_uploader):
        """Test returns None when Content-Type is missing"""
        mock_response = MagicMock()
        mock_response.headers = {
            "Content-Range": "bytes 0-0/1024",
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        uri, metadata = await pre_uploader._fetch_one_metadata_async(
            mock_client, "https://example.com/image.png"
        )

        assert uri == "https://example.com/image.png"
        assert metadata is None

    @staticmethod
    @pytest.mark.asyncio
    async def test_fetch_one_metadata_async_http_error(pre_uploader):
        """Test returns None on HTTP error"""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPError("Network error")
        )

        uri, metadata = await pre_uploader._fetch_one_metadata_async(
            mock_client, "https://example.com/image.png"
        )

        assert uri == "https://example.com/image.png"
        assert metadata is None

    # ========== Uri Successful Processing Tests ==========

    @staticmethod
    @patch.object(MultimodalPreUploader, "_fetch_metadata_batch")
    def test_uri_part_success_creates_download_upload_item(
        mock_fetch, pre_uploader
    ):
        """Test creating DOWNLOAD_AND_UPLOAD task after successful metadata fetching"""
        mock_fetch.return_value = {
            "https://example.com/image.png": UriMetadata(
                content_type="image/png",
                content_length=1024,
                etag='"abc123"',
                last_modified="Wed, 01 Jan 2020 00:00:00 GMT",
            )
        }

        part = Uri(
            uri="https://example.com/image.png",
            mime_type="image/png",
            modality="image",
        )
        message = InputMessage(role="user", parts=[part])

        input_messages = [message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 1
        assert uploads[0].data is None  # Download-upload task has no data
        assert uploads[0].source_uri == "https://example.com/image.png"
        assert uploads[0].expected_size == 1024
        assert uploads[0].content_type == "image/png"

        # Verify original part is replaced
        assert message.parts[0].uri != "https://example.com/image.png"
        assert message.parts[0].uri.endswith(".png")

    @staticmethod
    @patch.object(MultimodalPreUploader, "_fetch_metadata_batch")
    def test_uri_part_all_modalities(mock_fetch, pre_uploader):
        """Test all supported modalities (image/video/audio)"""
        test_cases = [
            ("image", "image/png", ".png"),
            ("video", "video/mp4", ".mp4"),
            ("audio", "audio/wav", ".wav"),
        ]

        for modality, content_type, ext in test_cases:
            uri = f"https://example.com/file{ext}"
            mock_fetch.return_value = {
                uri: UriMetadata(
                    content_type=content_type,
                    content_length=1024,
                    etag=f'"{modality}-etag"',
                )
            }

            part = Uri(uri=uri, mime_type=content_type, modality=modality)
            message = InputMessage(role="user", parts=[part])

            input_messages = [message]

            uploads = pre_uploader.pre_upload(
                span_context=None,
                start_time_utc_nano=1000000000000000000,
                input_messages=input_messages,
                output_messages=None,
            )

            assert len(uploads) == 1, f"Failed for modality {modality}"
            assert uploads[0].content_type == content_type
            assert uploads[0].source_uri == uri

    # ========== Uri Skip Scenarios Tests ==========

    @staticmethod
    @patch.object(MultimodalPreUploader, "_fetch_metadata_batch")
    def test_uri_part_skip_unsupported_modality(mock_fetch, pre_uploader):
        """Test unsupported modality doesn't call fetch"""
        part = Uri(
            uri="https://example.com/file.bin",
            mime_type="application/octet-stream",
            modality="__unsupported__",
        )
        message = InputMessage(role="user", parts=[part])

        input_messages = [message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 0
        # Unsupported modality should not call fetch (no URI to fetch)
        mock_fetch.assert_not_called()

    @staticmethod
    @patch.object(MultimodalPreUploader, "_fetch_metadata_batch")
    def test_uri_part_skip_metadata_fetch_failed(mock_fetch, pre_uploader):
        """Test keeping original when metadata fetch fails"""
        mock_fetch.return_value = {}  # No metadata returned

        original_uri = "https://example.com/fail.png"
        part = Uri(uri=original_uri, mime_type="image/png", modality="image")
        message = InputMessage(role="user", parts=[part])

        input_messages = [message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 0
        # Part should remain unchanged
        assert message.parts[0].uri == original_uri

    @staticmethod
    @patch.object(MultimodalPreUploader, "_fetch_metadata_batch")
    def test_uri_part_skip_size_exceeded(mock_fetch, pre_uploader):
        """Test keeping original when size exceeds limit"""
        mock_fetch.return_value = {
            "https://example.com/large.png": UriMetadata(
                content_type="image/png",
                content_length=_MAX_MULTIMODAL_DATA_SIZE + 1,  # Exceeds limit
                etag='"abc123"',
            )
        }

        original_uri = "https://example.com/large.png"
        part = Uri(uri=original_uri, mime_type="image/png", modality="image")
        message = InputMessage(role="user", parts=[part])

        input_messages = [message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 0
        # Part should remain unchanged
        assert message.parts[0].uri == original_uri


class TestPreUploadLimits:
    """Test multimodal data processing limits"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        return MultimodalPreUploader(base_path="/tmp/test_upload")

    @staticmethod
    def test_max_multimodal_parts_limit(pre_uploader):
        """Test processing at most _MAX_MULTIMODAL_PARTS parts"""
        test_data = b"test data"

        # Create more parts than the limit
        parts = []
        for _ in range(_MAX_MULTIMODAL_PARTS + 5):
            parts.append(
                Blob(
                    content=test_data, mime_type="image/png", modality="image"
                )
            )

        message = InputMessage(role="user", parts=parts)

        input_messages = [message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        # Process at most _MAX_MULTIMODAL_PARTS parts
        assert len(uploads) == _MAX_MULTIMODAL_PARTS

    @staticmethod
    @patch(
        "opentelemetry.util.genai._multimodal_upload.pre_uploader._MAX_MULTIMODAL_DATA_SIZE",
        100,
    )
    def test_blob_size_limit_exceeded(pre_uploader):
        """Test Blob larger than limit is skipped"""
        large_data = b"x" * 101
        part = Blob(
            content=large_data, mime_type="image/png", modality="image"
        )
        message = InputMessage(role="user", parts=[part])

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000,
            input_messages=[message],
            output_messages=[],
        )
        assert len(uploads) == 0

    @staticmethod
    @patch(
        "opentelemetry.util.genai._multimodal_upload.pre_uploader._MAX_MULTIMODAL_DATA_SIZE",
        100,
    )
    def test_base64_blob_size_limit_exceeded(pre_uploader):
        """Test Base64Blob larger than limit is skipped"""
        data = b"x" * 101
        b64_data = base64.b64encode(data).decode("ascii")
        part = Base64Blob(
            content=b64_data, mime_type="image/png", modality="image"
        )
        message = InputMessage(role="user", parts=[part])

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000,
            input_messages=[message],
            output_messages=[],
        )
        assert len(uploads) == 0

    @staticmethod
    @patch(
        "opentelemetry.util.genai._multimodal_upload.pre_uploader._MAX_MULTIMODAL_DATA_SIZE",
        100,
    )
    def test_data_uri_size_limit_exceeded(pre_uploader):
        """Test Data URI larger than limit is skipped"""
        data = b"x" * 101
        b64_data = base64.b64encode(data).decode("ascii")
        data_uri = f"data:image/png;base64,{b64_data}"

        part = Uri(modality="image", mime_type=None, uri=data_uri)
        message = InputMessage(role="user", parts=[part])

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000,
            input_messages=[message],
            output_messages=[],
        )
        assert len(uploads) == 0


class TestPreUploadEventLoop:
    """Test behavior in existing event loop scenarios"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        return MultimodalPreUploader(base_path="/tmp/test_upload")

    @staticmethod
    @pytest.mark.asyncio
    async def test_pre_upload_in_existing_event_loop(pre_uploader):
        """Test calling pre_upload to handle Uri in existing event loop"""
        with respx.mock:
            respx.get("https://example.com/test.png").mock(
                return_value=httpx.Response(
                    206,
                    headers={
                        "Content-Type": "image/png",
                        "Content-Range": "bytes 0-0/1024",
                        "ETag": '"test-etag"',
                    },
                    content=b"\x00",
                )
            )

            part = Blob(
                content=b"test data in async context",
                mime_type="image/png",
                modality="image",
            )
            message = InputMessage(role="user", parts=[part])

            input_messages = [message]

            test_data = b"test data in async context"

            part2 = Uri(
                uri="https://example.com/test.png",
                mime_type="image/png",
                modality="image",
            )
            message2 = InputMessage(role="assistant", parts=[part2])
            output_messages = [message2]

            uploads = pre_uploader.pre_upload(
                span_context=None,
                start_time_utc_nano=1000000000000000000,
                input_messages=input_messages,
                output_messages=output_messages,
            )

            assert len(uploads) == 2
            assert uploads[0].data == test_data
            assert uploads[1].source_uri == "https://example.com/test.png"

    @staticmethod
    def test_event_loop_is_instance_scoped():
        """Different pre-uploader instances should own independent loops."""
        pre_uploader_1 = MultimodalPreUploader(base_path="/tmp/test_upload_1")
        pre_uploader_2 = MultimodalPreUploader(base_path="/tmp/test_upload_2")

        loop_1 = pre_uploader_1._ensure_loop()
        loop_2 = pre_uploader_2._ensure_loop()

        assert loop_1 is not loop_2
        assert pre_uploader_1._loop_thread is not pre_uploader_2._loop_thread

        pre_uploader_1.shutdown(timeout=0.5)
        assert pre_uploader_1._loop is None
        assert pre_uploader_2._loop is not None

        pre_uploader_2.shutdown(timeout=0.5)

    @staticmethod
    def test_run_async_after_shutdown_returns_empty():
        """No new async task should be accepted after shutdown."""
        pre_uploader = MultimodalPreUploader(base_path="/tmp/test_upload")
        pre_uploader.shutdown(timeout=0.1)

        async def _dummy():
            await asyncio.sleep(0)
            return {
                "x": UriMetadata(content_type="image/png", content_length=1)
            }

        assert pre_uploader._run_async(_dummy(), timeout=0.1) == {}


class TestPreUploadNonHttpUri:
    """Test non-HTTP URI handling"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        return MultimodalPreUploader(base_path="/tmp/test_upload")

    @staticmethod
    @patch.object(MultimodalPreUploader, "_fetch_metadata_batch")
    def test_non_http_uri_skipped(mock_fetch, pre_uploader):
        """Test non-http and already processed URIs are not fetched again"""
        # Test non-http/https URI
        part = Uri(
            uri="file:///local/path/image.png",
            mime_type="image/png",
            modality="image",
        )
        message = InputMessage(role="user", parts=[part])

        input_messages = [message]
        # Simulate already processed URI (e.g., sls:// or other custom protocols)
        part = Uri(
            uri="sls://project/logstore/20241225/abc.png",
            mime_type="image/png",
            modality="image",
        )
        message2 = InputMessage(role="user", parts=[part])
        input_messages.append(message2)

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 0
        mock_fetch.assert_not_called()

    @staticmethod
    def test_is_http_uri_method(pre_uploader):
        """Test _is_http_uri method"""
        assert (
            MultimodalPreUploader._is_http_uri("http://example.com/test.png")
            is True
        )
        assert (
            MultimodalPreUploader._is_http_uri("https://example.com/test.png")
            is True
        )
        assert (
            MultimodalPreUploader._is_http_uri("HTTP://example.com/test.png")
            is False
        )  # Case sensitive
        assert (
            MultimodalPreUploader._is_http_uri("file:///local/path.png")
            is False
        )
        assert (
            MultimodalPreUploader._is_http_uri(
                "sls://project/logstore/file.png"
            )
            is False
        )
        assert MultimodalPreUploader._is_http_uri("/local/path.png") is False


class TestMultimodalUploadSwitch:
    """Test multimodal upload switch configuration

    Configuration is now read directly from environment variables, no longer depends on ArmsEnv singleton.
    """

    @staticmethod
    def test_upload_mode_none_skips_all():
        """Test OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE=none skips all processing"""
        with patch.dict(
            "os.environ",
            {"OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE": "none"},
        ):
            reset_multimodal_runtime_state_for_test()
            pre_uploader = MultimodalPreUploader("/tmp/test")
            assert pre_uploader._process_input is False
            assert pre_uploader._process_output is False

            # Even with multimodal data, don't process
            input_message = InputMessage(
                role="user",
                parts=[
                    Blob(
                        modality="image",
                        mime_type="image/png",
                        content=b"test",
                    )
                ],
            )
            input_messages = [input_message]

            uploads = pre_uploader.pre_upload(None, 0, input_messages, None)
            assert len(uploads) == 0

    @staticmethod
    def test_upload_mode_input_only():
        """OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE=input only processes input"""
        with patch.dict(
            "os.environ",
            {"OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE": "input"},
        ):
            reset_multimodal_runtime_state_for_test()
            pre_uploader = MultimodalPreUploader("/tmp/test")
            assert pre_uploader._process_input is True
            assert pre_uploader._process_output is False

    @staticmethod
    def test_upload_mode_output_only():
        """OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE=output only processes output"""
        with patch.dict(
            "os.environ",
            {"OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE": "output"},
        ):
            reset_multimodal_runtime_state_for_test()
            pre_uploader = MultimodalPreUploader("/tmp/test")
            assert pre_uploader._process_input is False
            assert pre_uploader._process_output is True

    @staticmethod
    def test_download_disabled_skips_uri():
        """OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED=false skips URI processing"""
        with patch.dict(
            "os.environ",
            {
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED": "false"
            },
        ):
            reset_multimodal_runtime_state_for_test()
            pre_uploader = MultimodalPreUploader("/tmp/test")
            assert pre_uploader._download_enabled is False

            # Blob still processes, Uri skipped
            input_message = InputMessage(
                role="user",
                parts=[
                    Blob(
                        modality="image",
                        mime_type="image/png",
                        content=b"test",
                    ),
                    Uri(
                        modality="image",
                        mime_type=None,
                        uri="https://example.com/img.jpg",
                    ),
                ],
            )
            input_messages = [input_message]

            uploads = pre_uploader.pre_upload(None, 0, input_messages, None)
            # Only processed Blob
            assert len(uploads) == 1
            assert uploads[0].data is not None  # Blob has data

            input_meta, output_meta = (
                MultimodalProcessingMixin._extract_multimodal_metadata(
                    input_messages, None
                )
            )
            assert not output_meta
            assert len(input_meta) == 2
            assert any(
                item.get("uri") == "https://example.com/img.jpg"
                for item in input_meta
            )
            uri_meta = next(
                item
                for item in input_meta
                if item.get("uri") == "https://example.com/img.jpg"
            )
            assert uri_meta.get("mime_type") == "image/jpeg"

    @staticmethod
    @patch.object(MultimodalPreUploader, "_fetch_metadata_batch")
    def test_download_enabled_fetch_failed_uri_kept_in_metadata(mock_fetch):
        """When metadata fetch fails, original URI should still appear in metadata."""
        with patch.dict(
            "os.environ",
            {
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE": "both",
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED": "true",
            },
        ):
            reset_multimodal_runtime_state_for_test()
            mock_fetch.return_value = {}
            pre_uploader = MultimodalPreUploader("/tmp/test")

            input_messages = [
                InputMessage(
                    role="user",
                    parts=[
                        Uri(
                            modality="image",
                            mime_type=None,
                            uri="https://example.com/fail.png",
                        )
                    ],
                )
            ]

            uploads = pre_uploader.pre_upload(None, 0, input_messages, None)
            assert not uploads

            input_meta, output_meta = (
                MultimodalProcessingMixin._extract_multimodal_metadata(
                    input_messages, None
                )
            )
            assert not output_meta
            assert len(input_meta) == 1
            assert input_meta[0]["uri"] == "https://example.com/fail.png"
            assert input_meta[0]["mime_type"] == "image/png"


class TestPreUploadDataUri:
    """Test data URI handling"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        """Create PreUploader instance"""
        return MultimodalPreUploader(
            base_path="/tmp/test_upload",
            extra_meta={"workspaceId": "test_workspace"},
        )

    @staticmethod
    def test_data_uri_processing(pre_uploader):
        """Test processing of base64 data URI"""
        # A small base64 image
        base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgDNjd8qAAAAAElFTkSuQmCC"
        data_uri = f"data:image/png;base64,{base64_data}"

        part = Uri(modality="image", mime_type=None, uri=data_uri)
        message = InputMessage(role="user", parts=[part])
        input_messages = [message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000,
            input_messages=input_messages,
            output_messages=[],
        )

        assert len(uploads) == 1
        assert uploads[0].content_type == "image/png"
        assert uploads[0].url.startswith("/tmp/test_upload")
        # Verify data is decoded correctly
        assert uploads[0].data == base64.b64decode(base64_data)
        # Verify original part is replaced with uploaded URL
        assert message.parts[0].uri != data_uri
        assert message.parts[0].uri == uploads[0].url

    @staticmethod
    def test_data_uri_processing_explicit_mime(pre_uploader):
        """Test processing of data URI with explicit mime type in Uri object"""
        base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgDNjd8qAAAAAElFTkSuQmCC"
        data_uri = f"data:image/png;base64,{base64_data}"

        part = Uri(modality="image", mime_type="image/custom", uri=data_uri)
        message = InputMessage(role="user", parts=[part])

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000,
            input_messages=[message],
            output_messages=[],
        )

        assert len(uploads) == 1
        assert uploads[0].content_type == "image/png"

    @staticmethod
    def test_invalid_data_uri(pre_uploader):
        """Test invalid data URI handling"""
        part = Uri(modality="image", mime_type=None, uri="data:invalid")
        message = InputMessage(role="user", parts=[part])
        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000,
            input_messages=[message],
            output_messages=[],
        )
        assert len(uploads) == 0

    @staticmethod
    def test_non_base64_data_uri_skipped(pre_uploader):
        """Test non-base64 data URI is skipped"""
        data_uri = "data:text/plain,hello%20world"
        part = Uri(modality="text", mime_type="text/plain", uri=data_uri)
        message = InputMessage(role="user", parts=[part])

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000,
            input_messages=[message],
            output_messages=[],
        )
        assert len(uploads) == 0
        assert message.parts[0].uri == data_uri


class TestPreUploadLocalFile:
    """Test local file handling with security checks"""

    @pytest.fixture
    def pre_uploader_factory(self):  # pylint: disable=R6301
        """Create PreUploader instance factory"""

        def _create():
            return MultimodalPreUploader(
                base_path="/tmp/test_upload",
                extra_meta={"workspaceId": "test_workspace"},
            )

        return _create

    @staticmethod
    def test_local_file_processing_allowed(pre_uploader_factory):
        """Test processing of local file URI when allowed"""
        # Use this test file itself as the source file
        test_file = Path(__file__).resolve()
        test_dir = test_file.parent
        file_uri = f"file://{test_file}"

        # Enable local file and set allowed root
        with patch.dict(
            os.environ,
            {
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_LOCAL_FILE_ENABLED": "true",
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_ALLOWED_ROOT_PATHS": str(
                    test_dir
                ),
            },
        ):
            reset_multimodal_runtime_state_for_test()
            pre_uploader = pre_uploader_factory()
            part = Uri(modality="image", mime_type="image/png", uri=file_uri)
            message = InputMessage(role="user", parts=[part])

            uploads = pre_uploader.pre_upload(
                span_context=None,
                start_time_utc_nano=1000000000000,
                input_messages=[message],
                output_messages=[],
            )

            assert len(uploads) == 1
            assert uploads[0].content_type == "image/png"
            assert uploads[0].url.startswith("/tmp/test_upload")
            # Verify data content matches file content
            assert uploads[0].data == test_file.read_bytes()
            # Verify original part is replaced
            assert message.parts[0].uri != file_uri
            assert message.parts[0].uri == uploads[0].url

    @staticmethod
    def test_local_file_processing_relative_path(pre_uploader_factory):
        """Test processing of relative path when allowed"""
        test_file = Path(__file__).resolve()
        test_dir = test_file.parent
        # Create a relative path: ./test_pre_uploader.py (assuming we run from same dir)
        # However, CWD might vary. Safer to use filename and rely on pre_uploader using CWD
        # Or construct a relative path if we know where we are.
        # Let's assume we allow the directory where this file resides.
        # And we pass the absolute path of the file but without scheme, which counts as local path.
        # OR we try to pass a relative path if we can force os.getcwd() to match.
        relative_path = test_file.name

        # Aliyun Python Agent Extension: for compatibility with Python 3.8
        with patch.dict(
                os.environ,
                {
                    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_LOCAL_FILE_ENABLED": "true",
                    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_ALLOWED_ROOT_PATHS": str(
                        test_dir
                    ),
                },
            ), \
            patch("os.getcwd", return_value=str(test_dir)):
            reset_multimodal_runtime_state_for_test()
            pre_uploader = pre_uploader_factory()
            # Test with simple filename (relative path)
            part = Uri(
                modality="image", mime_type="image/png", uri=relative_path
            )
            message = InputMessage(role="user", parts=[part])

            uploads = pre_uploader.pre_upload(
                span_context=None,
                start_time_utc_nano=1000000000000,
                input_messages=[message],
                output_messages=[],
            )

            assert len(uploads) == 1
            assert uploads[0].data == test_file.read_bytes()

    @staticmethod
    def test_local_file_processing_disabled_by_default(pre_uploader_factory):
        """Test local file ignored when disabled (default)"""
        test_file = Path(__file__).resolve()
        file_uri = f"file://{test_file}"

        # Default environment (feature disabled)
        pre_uploader = pre_uploader_factory()
        part = Uri(modality="image", mime_type="image/png", uri=file_uri)
        message = InputMessage(role="user", parts=[part])

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000,
            input_messages=[message],
            output_messages=[],
        )

        assert len(uploads) == 0
        assert message.parts[0].uri == file_uri

    @staticmethod
    def test_local_file_processing_forbidden_path(pre_uploader_factory):
        """Test blocked access when path is not in allowed roots"""
        test_file = Path(__file__).resolve()
        # Allowed root is /tmp, but file is in source dir
        allowed_root = "/tmp/fake_allowed_root"

        with patch.dict(
            os.environ,
            {
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_LOCAL_FILE_ENABLED": "true",
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_ALLOWED_ROOT_PATHS": allowed_root,
            },
        ):
            reset_multimodal_runtime_state_for_test()
            pre_uploader = pre_uploader_factory()
            file_uri = f"file://{test_file}"
            part = Uri(modality="image", mime_type="image/png", uri=file_uri)
            message = InputMessage(role="user", parts=[part])

            uploads = pre_uploader.pre_upload(
                span_context=None,
                start_time_utc_nano=1000,
                input_messages=[message],
                output_messages=[],
            )

            assert len(uploads) == 0
            # URI should remain unchanged
            assert message.parts[0].uri == file_uri

    @staticmethod
    def test_local_file_processing_symlink_traversal_blocked(
        pre_uploader_factory, tmp_path
    ):
        """Test that symlink traversal outside allowed root is blocked"""
        allowed_root = tmp_path / "allowed_root"
        external_dir = tmp_path / "outside_root"
        allowed_root.mkdir()
        external_dir.mkdir()

        secret_file = external_dir / "secret.txt"
        secret_file.write_text("top secret", encoding="utf-8")

        symlink_dir = allowed_root / "symlink_to_outside"
        try:
            os.symlink(external_dir, symlink_dir)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks are not supported on this platform")

        target_path = symlink_dir / "secret.txt"
        file_uri = f"file://{target_path}"

        with patch.dict(
            os.environ,
            {
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_LOCAL_FILE_ENABLED": "true",
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_ALLOWED_ROOT_PATHS": str(
                    allowed_root
                ),
            },
        ):
            reset_multimodal_runtime_state_for_test()
            pre_uploader = pre_uploader_factory()
            part = Uri(modality="image", mime_type="image/png", uri=file_uri)
            message = InputMessage(role="user", parts=[part])

            uploads = pre_uploader.pre_upload(
                span_context=None,
                start_time_utc_nano=1000,
                input_messages=[message],
                output_messages=[],
            )

        assert len(uploads) == 0
        assert message.parts[0].uri == file_uri

    @staticmethod
    def test_local_file_processing_no_allowed_roots(pre_uploader_factory):
        """Test blocked access when no allowed roots configured"""
        test_file = Path(__file__).resolve()

        with patch.dict(
            os.environ,
            {
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_LOCAL_FILE_ENABLED": "true",
                # No ALLOWED_ROOT_PATHS
            },
        ):
            reset_multimodal_runtime_state_for_test()
            pre_uploader = pre_uploader_factory()
            file_uri = f"file://{test_file}"
            part = Uri(modality="image", mime_type="image/png", uri=file_uri)
            message = InputMessage(role="user", parts=[part])

            uploads = pre_uploader.pre_upload(
                span_context=None,
                start_time_utc_nano=1000,
                input_messages=[message],
                output_messages=[],
            )

            assert len(uploads) == 0
