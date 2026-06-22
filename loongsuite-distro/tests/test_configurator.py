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

from loongsuite.distro import LoongSuiteConfigurator
from loongsuite.distro.resource import (
    GEN_AI_INSTRUMENTATION_SDK_NAME,
    HOST_IP,
    SERVICE_INSTANCE_ID,
)


class TestLoongSuiteConfigurator(unittest.TestCase):
    @mock.patch("opentelemetry.sdk._configuration._initialize_components")
    def test_configure_injects_loongsuite_attributes(self, mock_init):
        LoongSuiteConfigurator().configure()

        mock_init.assert_called_once()
        resource_attributes = mock_init.call_args.kwargs["resource_attributes"]
        self.assertIn(HOST_IP, resource_attributes)
        self.assertIn(SERVICE_INSTANCE_ID, resource_attributes)
        self.assertEqual(
            resource_attributes[GEN_AI_INSTRUMENTATION_SDK_NAME],
            "loongsuite-genai-utils",
        )

    @mock.patch("opentelemetry.sdk._configuration._initialize_components")
    def test_configure_preserves_existing_attributes(self, mock_init):
        """User-provided values take precedence over detector values."""
        LoongSuiteConfigurator().configure(
            resource_attributes={
                "service.name": "my-service",
                HOST_IP: "custom-ip",
            }
        )

        resource_attributes = mock_init.call_args.kwargs["resource_attributes"]
        self.assertEqual(resource_attributes["service.name"], "my-service")
        # User-provided host.ip should NOT be overwritten by detector
        self.assertEqual(resource_attributes[HOST_IP], "custom-ip")
        self.assertIn(SERVICE_INSTANCE_ID, resource_attributes)
        self.assertIn(GEN_AI_INSTRUMENTATION_SDK_NAME, resource_attributes)

    @mock.patch("opentelemetry.sdk._configuration._initialize_components")
    def test_configure_forwards_other_kwargs(self, mock_init):
        LoongSuiteConfigurator().configure(
            auto_instrumentation_version="1.2.3"
        )

        self.assertEqual(
            mock_init.call_args.kwargs["auto_instrumentation_version"],
            "1.2.3",
        )


if __name__ == "__main__":
    unittest.main()
