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

"""Client for the ARMS presigned multimodal upload URL API.

In the pre-authorized OSS mode the agent never holds OSS credentials: it asks
ARMS for a short-lived presigned URL (authenticated with the agent license
key) and then writes the object straight to the ARMS-owned bucket.

The API answers with the presigned URL as a bare text body; JSON envelopes are
also accepted so the client keeps working if the API starts wrapping the URL.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import httpx

from opentelemetry.instrumentation.utils import suppress_http_instrumentation

_logger = logging.getLogger(__name__)

PRESIGN_API_PATH = "/apm/meta/api/v1/multimodal/upload/presign"
LICENSE_KEY_HEADER = "x-arms-license-key"
WORKSPACE_HEADER = "x-cms-workspace"

_URL_KEYS = (
    "uploadUrl",
    "upload_url",
    "uploadURL",
    "signedUrl",
    "signed_url",
    "presignUrl",
    "presignedUrl",
    "presigned_url",
    "url",
)
_METHOD_KEYS = ("method", "httpMethod", "http_method", "uploadMethod")
_HEADER_KEYS = ("headers", "requestHeaders", "signedHeaders")
_EXPIRATION_KEYS = (
    "expiration",
    "expireTime",
    "expiredTime",
    "expires",
    "expireAt",
)
_ENVELOPE_KEYS = ("data", "result", "Data", "Result", "body")
_SUCCESS_KEYS = ("success", "Success")
_FAILURE_MESSAGE_KEYS = ("message", "Message", "errorMessage", "msg")


class PresignError(Exception):
    """Base error raised while requesting a presigned upload URL."""


class PresignConfigError(PresignError, ValueError):
    """Presign endpoint, credentials, or target object are not usable."""


class PresignAuthError(PresignError):
    """ARMS rejected the license key or workspace."""


class PresignRetryableError(PresignError):
    """Transient failure: the request may succeed on a later attempt."""


@dataclass(frozen=True)
class PresignedUpload:
    """A presigned target the agent can write one object to."""

    url: str
    method: str = "PUT"
    headers: Mapping[str, str] = field(default_factory=dict)
    expiration: Optional[str] = None


def resolve_presign_endpoint(snapshot: Any) -> Optional[str]:
    """Resolve the presign endpoint from config, then from ARMS state.

    The ARMS OneEndpoint state is imported lazily and treated as optional so
    that the multimodal pipeline keeps working in community deployments where
    the ARMS SDK extension is absent.
    """
    configured = getattr(snapshot, "presign_endpoint", None)
    if configured:
        return configured
    try:
        from aliyun.sdk.extension.arms.exporters.arms_endpoints_state import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
            global_arms_endpoints_state,
        )

        endpoint = global_arms_endpoints_state.get_one_endpoint()
    except Exception:  # pylint: disable=broad-except
        _logger.debug(
            "ARMS endpoint state unavailable for multimodal presign",
            exc_info=True,
        )
        return None
    return endpoint or None


def _unwrap_payload(payload: Any) -> Optional[Mapping[str, Any]]:
    """Walk known envelope keys until a mapping carrying a URL is found."""
    seen = 0
    current = payload
    while isinstance(current, Mapping) and seen < 4:
        if any(current.get(key) for key in _URL_KEYS):
            return current
        for key in _ENVELOPE_KEYS:
            nested = current.get(key)
            if isinstance(nested, Mapping):
                current = nested
                break
        else:
            return current
        seen += 1
    return current if isinstance(current, Mapping) else None


def _envelope_failure(payload: Any) -> Optional[str]:
    if not isinstance(payload, Mapping):
        return None
    for key in _SUCCESS_KEYS:
        if key in payload and payload[key] is False:
            for message_key in _FAILURE_MESSAGE_KEYS:
                message = payload.get(message_key)
                if message:
                    return str(message)
            return "presign request reported failure"
    return None


def _string_headers(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): str(header_value)
        for name, header_value in value.items()
        if name and header_value is not None
    }


def _plain_url(text: str) -> Optional[str]:
    """Return ``text`` when it carries a single bare http(s) URL."""
    candidate = (text or "").strip()
    if len(candidate) > 1 and candidate[0] == '"' and candidate[-1] == '"':
        candidate = candidate[1:-1].strip()
    if not candidate or len(candidate.split()) > 1:
        return None
    if candidate.lower().startswith(("http://", "https://")):
        return candidate
    return None


def parse_presign_response(payload: Any) -> PresignedUpload:
    """Parse a presign response, tolerating envelope and naming variants."""
    failure = _envelope_failure(payload)
    if failure:
        raise PresignError(f"Presign request failed: {failure}")

    body = _unwrap_payload(payload)
    if body is None:
        raise PresignError("Presign response is not a JSON object")

    url = ""
    for key in _URL_KEYS:
        candidate = body.get(key)
        if candidate:
            url = str(candidate).strip()
            break
    if not url:
        raise PresignError("Presign response does not contain an upload URL")

    method = "PUT"
    for key in _METHOD_KEYS:
        candidate = body.get(key)
        if candidate:
            method = str(candidate).strip().upper()
            break

    headers: Dict[str, str] = {}
    for key in _HEADER_KEYS:
        headers = _string_headers(body.get(key))
        if headers:
            break

    expiration: Optional[str] = None
    for key in _EXPIRATION_KEYS:
        candidate = body.get(key)
        if candidate is not None and str(candidate).strip():
            expiration = str(candidate).strip()
            break

    return PresignedUpload(
        url=url,
        method=method or "PUT",
        headers=headers,
        expiration=expiration,
    )


def parse_presign_body(text: str) -> PresignedUpload:
    """Parse a presign response body holding a bare URL or a JSON envelope."""
    url = _plain_url(text)
    if url:
        return PresignedUpload(url=url)
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise PresignError(
            f"Presign response is neither a URL nor JSON: {exc}"
        ) from exc
    if isinstance(payload, str):
        url = _plain_url(payload)
        if not url:
            raise PresignError(
                "Presign response does not contain an upload URL"
            )
        return PresignedUpload(url=url)
    return parse_presign_response(payload)


class MultimodalPresignClient:
    """Requests presigned OSS upload URLs from the ARMS meta API."""

    def __init__(
        self,
        *,
        license_key: str,
        workspace: str = "",
        project: str = "",
        logstore: str = "",
        endpoint: Optional[str] = None,
        endpoint_provider: Optional[Any] = None,
        timeout: float = 30.0,
        verify: bool = True,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._license_key = (license_key or "").strip()
        if not self._license_key:
            raise PresignConfigError(
                "Multimodal presign requires an ARMS license key"
            )
        if not endpoint and endpoint_provider is None:
            raise PresignConfigError(
                "Multimodal presign requires an endpoint or endpoint provider"
            )
        self._workspace = (workspace or "").strip()
        self._project = (project or "").strip()
        self._logstore = (logstore or "").strip()
        self._endpoint = endpoint
        self._endpoint_provider = endpoint_provider
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.Client(
            timeout=timeout,
            verify=verify,
            follow_redirects=False,
        )

    @property
    def project(self) -> str:
        return self._project

    @property
    def logstore(self) -> str:
        return self._logstore

    def _presign_url(self) -> str:
        endpoint = self._endpoint
        if not endpoint and self._endpoint_provider is not None:
            endpoint = self._endpoint_provider()
        if not endpoint:
            raise PresignRetryableError(
                "Multimodal presign endpoint is not resolvable yet"
            )
        return f"{endpoint}{PRESIGN_API_PATH}"

    def _request_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            LICENSE_KEY_HEADER: self._license_key,
        }
        if self._workspace:
            headers[WORKSPACE_HEADER] = self._workspace
        return headers

    def build_payload(self, object_name: str) -> Dict[str, str]:
        key = (object_name or "").lstrip("/")
        if not key:
            raise PresignConfigError("Presign object name is required")
        # No bucket field: ARMS owns the backing bucket and ignores any bucket
        # the agent sends, so advertising one here would only be misleading.
        payload = {"objectName": key}
        if self._project:
            payload["project"] = self._project
        if self._logstore:
            payload["logstore"] = self._logstore
        return payload

    def presign(self, object_name: str) -> PresignedUpload:
        """Return a presigned target for ``object_name``.

        Raises:
            PresignConfigError: the object name is unusable.
            PresignAuthError: ARMS rejected the license key or workspace.
            PresignRetryableError: transient network or server-side failure.
            PresignError: the response could not be used.
        """
        url = self._presign_url()
        payload = self.build_payload(object_name)
        with suppress_http_instrumentation():
            try:
                response = self._http_client.post(
                    url,
                    headers=self._request_headers(),
                    content=json.dumps(payload).encode("utf-8"),
                )
            except httpx.TransportError as exc:
                raise PresignRetryableError(
                    f"Presign request transport error: {exc}"
                ) from exc

        if response.status_code in (401, 403):
            raise PresignAuthError(
                f"Presign request unauthorized (status={response.status_code})"
            )
        if response.status_code in (408, 429) or response.status_code >= 500:
            raise PresignRetryableError(
                f"Presign request failed (status={response.status_code})"
            )
        if not response.is_success:
            raise PresignError(
                f"Presign request rejected (status={response.status_code})"
            )

        return parse_presign_body(response.text)

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()
