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
import socket
from logging import getLogger

from opentelemetry.sdk.resources import Resource, ResourceDetector

logger = getLogger(__name__)

# Resource attribute keys contributed by LoongSuite.
HOST_IP = "host.ip"
SERVICE_INSTANCE_ID = "service.instance.id"
GEN_AI_INSTRUMENTATION_SDK_NAME = "gen_ai.instrumentation.sdk.name"

# Fixed value identifying the GenAI instrumentation SDK shipped with LoongSuite.
_GEN_AI_INSTRUMENTATION_SDK_NAME_VALUE = "loongsuite-genai-utils"

_FALLBACK_HOST_IP = "127.0.0.1"


def _get_host_ip() -> str:
    """Best-effort detection of the local host IP address.

    Opens a UDP socket towards a public address to discover which local
    interface would be used for outbound traffic. No packet is actually sent.
    Falls back to ``127.0.0.1`` when detection fails.
    """
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError as exception:
        logger.debug(
            "Failed to detect host ip, falling back to %s. Exception: %s",
            _FALLBACK_HOST_IP,
            exception,
        )
        return _FALLBACK_HOST_IP
    finally:
        if sock is not None:
            sock.close()


def _get_host_ip_with_pid() -> str:
    """Returns the host IP combined with the current process id.

    The format is ``<ip>-<pid>``, for example ``127.0.0.1-1``.
    """
    return f"{_get_host_ip()}-{os.getpid()}"


class LoongSuiteResourceDetector(ResourceDetector):
    """Detects LoongSuite specific resource attributes.

    Contributes the following attributes to the resource:

    * ``host.ip`` — the raw host IP address (e.g. ``127.0.0.1``).
    * ``service.instance.id`` — ``<ip>-<pid>`` uniquely identifying the process
      instance (e.g. ``127.0.0.1-1``).
    * ``gen_ai.instrumentation.sdk.name`` set to ``loongsuite-genai-utils``.
    """

    def detect(self) -> Resource:
        host_ip = _get_host_ip()
        return Resource(
            {
                HOST_IP: host_ip,
                SERVICE_INSTANCE_ID: f"{host_ip}-{os.getpid()}",
                GEN_AI_INSTRUMENTATION_SDK_NAME: _GEN_AI_INSTRUMENTATION_SDK_NAME_VALUE,
            }
        )
