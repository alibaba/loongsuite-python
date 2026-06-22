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

import unittest
from unittest import mock

from loongsuite.distro.resource import (
    GEN_AI_INSTRUMENTATION_SDK_NAME,
    HOST_IP,
    SERVICE_INSTANCE_ID,
    LoongSuiteResourceDetector,
    _get_host_ip,
    _get_host_ip_with_pid,
)

from opentelemetry.sdk.resources import Resource, ResourceDetector


class TestLoongSuiteResourceDetector(unittest.TestCase):
    def test_is_resource_detector(self):
        self.assertIsInstance(LoongSuiteResourceDetector(), ResourceDetector)

    def test_detect_returns_resource(self):
        resource = LoongSuiteResourceDetector().detect()
        self.assertIsInstance(resource, Resource)

    def test_detect_contains_expected_keys(self):
        attributes = LoongSuiteResourceDetector().detect().attributes
        self.assertIn(HOST_IP, attributes)
        self.assertIn(SERVICE_INSTANCE_ID, attributes)
        self.assertIn(GEN_AI_INSTRUMENTATION_SDK_NAME, attributes)

    def test_host_ip_is_raw_ip(self):
        """host.ip should contain a raw IP address without PID suffix."""
        attributes = LoongSuiteResourceDetector().detect().attributes
        # Raw IP should not contain a dash-PID suffix
        self.assertNotIn("-", attributes[HOST_IP].rsplit(".", 1)[-1])

    def test_gen_ai_instrumentation_sdk_name_value(self):
        attributes = LoongSuiteResourceDetector().detect().attributes
        self.assertEqual(
            attributes[GEN_AI_INSTRUMENTATION_SDK_NAME],
            "loongsuite-genai-utils",
        )

    @mock.patch("loongsuite.distro.resource.os.getpid", return_value=1)
    @mock.patch(
        "loongsuite.distro.resource._get_host_ip", return_value="127.0.0.1"
    )
    def test_service_instance_id_format_is_ip_dash_pid(
        self, _mock_ip, _mock_pid
    ):
        attributes = LoongSuiteResourceDetector().detect().attributes
        self.assertEqual(attributes[SERVICE_INSTANCE_ID], "127.0.0.1-1")
        self.assertEqual(attributes[HOST_IP], "127.0.0.1")

    @mock.patch("loongsuite.distro.resource.os.getpid", return_value=42)
    @mock.patch(
        "loongsuite.distro.resource._get_host_ip", return_value="10.0.0.5"
    )
    def test_get_host_ip_with_pid(self, _mock_ip, _mock_pid):
        self.assertEqual(_get_host_ip_with_pid(), "10.0.0.5-42")


class TestGetHostIp(unittest.TestCase):
    def test_returns_detected_ip(self):
        mock_sock = mock.MagicMock()
        mock_sock.getsockname.return_value = ("192.168.1.100", 12345)
        with mock.patch(
            "loongsuite.distro.resource.socket.socket",
            return_value=mock_sock,
        ):
            self.assertEqual(_get_host_ip(), "192.168.1.100")
        mock_sock.connect.assert_called_once()
        mock_sock.close.assert_called_once()

    def test_falls_back_to_loopback_on_error(self):
        mock_sock = mock.MagicMock()
        mock_sock.connect.side_effect = OSError("no network")
        with mock.patch(
            "loongsuite.distro.resource.socket.socket",
            return_value=mock_sock,
        ):
            self.assertEqual(_get_host_ip(), "127.0.0.1")
        # The socket must still be closed even when detection fails.
        mock_sock.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
