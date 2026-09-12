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

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional, Tuple
from urllib.parse import urlsplit


def get_client_info(instance: Any) -> Tuple[bool, Optional[str]]:
    is_vertex = False
    server_address = None

    if hasattr(instance, "_api_client"):
        api_client = instance._api_client
        is_vertex = getattr(api_client, "vertexai", False)
        if hasattr(api_client, "_http_options"):
            server_address = getattr(
                api_client._http_options, "base_url", None
            )
    elif hasattr(instance, "_client"):
        client = instance._client
        is_vertex = getattr(client, "_is_vertex", False)
        server_address = getattr(client, "server", None)
    elif hasattr(instance, "sdk_configuration"):
        config = instance.sdk_configuration
        server_url = getattr(config, "server_url", "")
        if server_url:
            server_address = server_url
            server_url_string = str(server_url)
            if "://" not in server_url_string:
                server_url_string = "//" + server_url_string
            server_hostname = urlsplit(server_url_string).hostname
            if server_hostname and (
                server_hostname == "aiplatform.googleapis.com"
                or server_hostname.endswith("-aiplatform.googleapis.com")
            ):
                is_vertex = True

    if server_address and "://" in str(server_address):
        server_address = urlsplit(str(server_address)).hostname
    elif server_address:
        server_address = str(server_address).rstrip("/")

    if not server_address:
        server_address = (
            "aiplatform.googleapis.com"
            if is_vertex
            else "generativelanguage.googleapis.com"
        )

    return bool(is_vertex), server_address
