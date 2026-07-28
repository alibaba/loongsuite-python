# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit


def get_client_info(instance: Any) -> tuple[bool, str | None]:
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
            if "aiplatform.googleapis.com" in server_url:
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
