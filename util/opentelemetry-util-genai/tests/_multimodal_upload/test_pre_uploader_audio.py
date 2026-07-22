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
Test automatic audio format detection for MultimodalPreUploader
Focuses on testing the _detect_audio_format method's ability to recognize various audio formats
and audio format conversion (e.g., PCM16 to WAV)
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from opentelemetry.util.genai._multimodal_upload import MultimodalPreUploader
from opentelemetry.util.genai._multimodal_upload.pre_uploader import (
    _audio_libs_available,
)
from opentelemetry.util.genai.types import Blob, InputMessage

from .multimodal_test_helpers import reset_multimodal_runtime_state_for_test

# Test audio file directory
TEST_AUDIO_DIR = Path(__file__).parent / "test_audio_samples"


@pytest.fixture(autouse=True)
def _default_upload_mode_enabled_for_tests():
    with patch.dict(
        "os.environ",
        {
            "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE": "both",
            "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED": "true",
            "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_AUDIO_CONVERSION_ENABLED": "true",
        },
    ):
        reset_multimodal_runtime_state_for_test()
        yield


class TestAudioFormatDetection:
    """Test automatic audio format detection functionality"""

    @pytest.fixture
    def pre_uploader(self):  # pylint: disable=R6301
        """Create PreUploader instance"""
        base_path = "/tmp/test_upload"
        return MultimodalPreUploader(base_path=base_path)

    @staticmethod
    def _read_audio_file(filename: str) -> bytes:
        """Read audio file"""
        filepath = TEST_AUDIO_DIR / filename
        if not filepath.exists():
            pytest.skip(f"Test audio file does not exist: {filepath}")

        with open(filepath, "rb") as file_obj:
            return file_obj.read()

    @staticmethod
    def test_import_without_audio_libs_does_not_write_to_standard_streams():
        """Missing optional audio libs should not emit import-time output."""
        project_root = Path(__file__).parents[4]
        util_genai_src = Path(__file__).parents[2] / "src"
        instrumentation_src = (
            project_root / "opentelemetry-instrumentation" / "src"
        )
        env = os.environ.copy()
        pythonpath_parts = [
            str(util_genai_src),
            str(instrumentation_src),
            env.get("PYTHONPATH", ""),
        ]
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in pythonpath_parts if part
        )

        script = textwrap.dedent(
            """
            import importlib.abc
            import logging
            import sys

            class BlockAudioLibs(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "numpy" or fullname.startswith("numpy."):
                        raise ImportError(fullname)
                    if fullname == "soundfile" or fullname.startswith("soundfile."):
                        raise ImportError(fullname)
                    return None

            sys.meta_path.insert(0, BlockAudioLibs())
            logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
            import opentelemetry.util.genai._multimodal_upload.pre_uploader  # noqa: F401
            """
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        assert completed.stdout == ""
        assert completed.stderr == ""

    # ========== Edge Case Tests ==========

    @staticmethod
    def test_detect_unknown_format(pre_uploader):
        """Test unrecognizable format"""
        unknown_data = b"\x00" * 100
        detected = pre_uploader._detect_audio_format(unknown_data)
        assert detected is None

    @staticmethod
    def test_detect_insufficient_data(pre_uploader):
        """Test insufficient data case"""
        short_data = b"\x00" * 5
        detected = pre_uploader._detect_audio_format(short_data)
        assert detected is None

    # ========== Real Audio File Format Detection Tests ==========

    @staticmethod
    @pytest.mark.parametrize(
        "filename,expected_mime,file_header_check",
        [
            (
                "test.wav",
                "audio/wav",
                lambda d: d[:4] == b"RIFF" and d[8:12] == b"WAVE",
            ),
            (
                "test.mp3",
                "audio/mp3",
                lambda d: d[:3] == b"ID3"
                or (d[0] == 0xFF and (d[1] & 0xE0) == 0xE0),
            ),
            (
                "test.aac",
                "audio/aac",
                lambda d: d[0] == 0xFF and (d[1] & 0xF6) == 0xF0,
            ),
            ("test.m4a", "audio/m4a", lambda d: d[4:8] == b"ftyp"),
            ("test.ogg", "audio/ogg", lambda d: d[:4] == b"OggS"),
            ("test.flac", "audio/flac", lambda d: d[:4] == b"fLaC"),
            ("test.amr", "audio/amr", lambda d: d[:6] == b"#!AMR\n"),
            ("test.3gp", "audio/3gpp", lambda d: d[4:8] == b"ftyp"),
        ],
    )
    def test_detect_real_audio_format(
        pre_uploader, filename, expected_mime, file_header_check
    ):
        """Test real audio file format detection"""
        data = TestAudioFormatDetection._read_audio_file(filename)

        # Verify file header
        assert file_header_check(data), f"{filename} file header is incorrect"

        # Test format detection
        detected = pre_uploader._detect_audio_format(data)
        # AMR and 3GP may have multiple MIME types
        if expected_mime == "audio/amr":
            assert detected in ("audio/amr", "audio/amr-wb"), (
                f"{filename} format detection failed, detected: {detected}"
            )
        elif expected_mime == "audio/3gpp":
            assert detected in ("audio/3gpp", "audio/3gpp2"), (
                f"{filename} format detection failed, detected: {detected}"
            )
        else:
            assert detected == expected_mime, (
                f"{filename} format detection failed, expected {expected_mime}, got {detected}"
            )

    # ========== PCM16 to WAV Conversion Tests ==========

    @staticmethod
    @pytest.mark.parametrize(
        "pcm_mime_type",
        [
            "audio/pcm16",
            "audio/l16",
            "audio/pcm",
        ],
    )
    def test_pcm16_to_wav_conversion(pre_uploader, pcm_mime_type):
        """Test PCM16 to WAV format conversion"""
        # Create simulated PCM16 data
        pcm_data = b"\x00\x01" * 1000

        part = Blob(
            content=pcm_data, mime_type=pcm_mime_type, modality="audio"
        )

        input_message = InputMessage(role="user", parts=[part])
        input_messages = [input_message]

        uploads = pre_uploader.pre_upload(
            span_context=None,
            start_time_utc_nano=1000000000000000000,
            input_messages=input_messages,
            output_messages=None,
        )

        assert len(uploads) == 1
        # Verify result based on library availability
        if _audio_libs_available:
            assert uploads[0].content_type == "audio/wav"
            assert uploads[0].url.endswith(".wav")
        else:
            # If library unavailable, should keep original format
            assert uploads[0].content_type == pcm_mime_type

    @staticmethod
    def test_pcm16_conversion_missing_audio_libs_logs_single_warning(
        caplog,
    ):
        """Missing optional audio libs should only log the actual conversion skip."""
        with patch(
            "opentelemetry.util.genai._multimodal_upload.pre_uploader._audio_libs_available",
            False,
        ), patch(
            "opentelemetry.util.genai._multimodal_upload.pre_uploader.np",
            None,
        ), patch(
            "opentelemetry.util.genai._multimodal_upload.pre_uploader.sf",
            None,
        ):
            pre_uploader = MultimodalPreUploader(base_path="/tmp/test_upload")
            part = Blob(
                content=b"\x00\x01" * 1000,
                mime_type="audio/pcm16",
                modality="audio",
            )
            input_messages = [InputMessage(role="user", parts=[part])]

            with caplog.at_level(
                "WARNING",
                logger=(
                    "opentelemetry.util.genai._multimodal_upload.pre_uploader"
                ),
            ):
                uploads = pre_uploader.pre_upload(
                    span_context=None,
                    start_time_utc_nano=1000000000000000000,
                    input_messages=input_messages,
                    output_messages=None,
                )

        assert len(uploads) == 1
        assert uploads[0].content_type == "audio/pcm16"
        warning_messages = [record.getMessage() for record in caplog.records]
        assert warning_messages == [
            "Failed to convert PCM16 to WAV, using original format"
        ]

    @staticmethod
    def test_pcm16_conversion_disabled_by_default():
        """Test PCM16 conversion stays disabled when env var is unset"""
        with patch.dict(
            "os.environ",
            {
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE": "both",
                "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED": "true",
            },
            clear=True,
        ):
            reset_multimodal_runtime_state_for_test()
            pre_uploader = MultimodalPreUploader(base_path="/tmp/test_upload")
            pcm_data = b"\x00\x01" * 1000
            part = Blob(
                content=pcm_data, mime_type="audio/pcm16", modality="audio"
            )
            input_messages = [InputMessage(role="user", parts=[part])]

            uploads = pre_uploader.pre_upload(
                span_context=None,
                start_time_utc_nano=1000000000000000000,
                input_messages=input_messages,
                output_messages=None,
            )

            assert len(uploads) == 1
            assert uploads[0].content_type == "audio/pcm16"
