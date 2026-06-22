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

import os
from typing import Any

from loongsuite.distro.resource import LoongSuiteResourceDetector

from opentelemetry.environment_variables import (
    OTEL_LOGS_EXPORTER,
    OTEL_METRICS_EXPORTER,
    OTEL_TRACES_EXPORTER,
)
from opentelemetry.instrumentation.distro import BaseDistro
from opentelemetry.sdk._configuration import _OTelSDKConfigurator
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_PROTOCOL


class LoongSuiteConfigurator(_OTelSDKConfigurator):
    """
    LoongSuite configurator, inherits from OpenTelemetry SDK configurator.

    Augments the resource with LoongSuite specific attributes (``host.ip`` and
    ``gen_ai.instrumentation.sdk.name``) before delegating to the OpenTelemetry
    SDK configurator.
    """

    def _configure(self, **kwargs: Any) -> None:
        resource_attributes = dict(
            kwargs.pop("resource_attributes", None) or {}
        )
        # Use setdefault so user-provided values take precedence over detector.
        detected = LoongSuiteResourceDetector().detect().attributes
        for k, v in detected.items():
            resource_attributes.setdefault(k, v)
        super()._configure(resource_attributes=resource_attributes, **kwargs)


class LoongSuiteDistro(BaseDistro):
    """
    LoongSuite Distro configures default OpenTelemetry settings.

    This is the Distro provided by LoongSuite, which configures default exporters and protocols.
    """

    # pylint: disable=no-self-use
    def _configure(self, **kwargs: Any) -> None:
        os.environ.setdefault(OTEL_TRACES_EXPORTER, "otlp")
        os.environ.setdefault(OTEL_METRICS_EXPORTER, "otlp")
        os.environ.setdefault(OTEL_LOGS_EXPORTER, "otlp")
        os.environ.setdefault(OTEL_EXPORTER_OTLP_PROTOCOL, "grpc")
