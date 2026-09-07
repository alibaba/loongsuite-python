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

"""Tests for the pre-authorized OSS (presigned URL) multimodal uploader."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import pytest

from opentelemetry.util.genai._multimodal_upload import UploadItem
from opentelemetry.util.genai._multimodal_upload.config import (  # pylint: disable=no-name-in-module
    DEFAULT_SLS_LOGSTORE,
    PRESIGN_HOOK_NAME,
    UPLOADER_GENERATION_FIELDS,
    format_sls_base_path,
    get_multimodal_config_snapshot,
    normalize_multimodal_hook_name,
    normalize_oss_bucket,
    normalize_oss_path_prefix,
    update_multimodal_runtime_config,
)
from opentelemetry.util.genai._multimodal_upload.presign_client import (  # pylint: disable=no-name-in-module
    LICENSE_KEY_HEADER,
    PRESIGN_API_PATH,
    WORKSPACE_HEADER,
    MultimodalPresignClient,
    PresignAuthError,
    PresignConfigError,
    PresignError,
    PresignRetryableError,
    parse_presign_response,
    resolve_presign_endpoint,
)
from opentelemetry.util.genai._multimodal_upload.presign_uploader import (  # pylint: disable=no-name-in-module
    PresignUploader,
    _presign_timeout,
    _resolve_sls_target,
    build_presign_client,
    presign_pre_uploader_hook,
    presign_uploader_hook,
)
from opentelemetry.util.genai._multimodal_upload.usage_recorder import (
    set_multimodal_usage_recorder,
)
from opentelemetry.util.genai.extended_environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_BUCKET,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_PATH_PREFIX,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_ENDPOINT,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_LICENSE_KEY,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_TIMEOUT,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_WORKSPACE,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER,
)

from .multimodal_test_helpers import reset_multimodal_runtime_state_for_test

_ENDPOINT = "https://proj-xtrace-test.cn-hangzhou.log.aliyuncs.com"
# The signed URL points at an ARMS-owned bucket the agent never configures.
_SIGNED_URL = (
    "https://bucket-a.oss-cn-hangzhou.aliyuncs.com/genai/img.jpg?sig=1"
)
_SLS_PROJECT_ENV = "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_PROJECT"
_SLS_LOGSTORE_ENV = "APSARA_APM_COLLECTOR_MULTIMODAL_SLS_LOGSTORE"
_PROJECT = "proj-xtrace-test"
_LOGSTORE = "logstore-multimodal"
_BASE_PATH = f"sls://{_PROJECT}/{_LOGSTORE}"
_UPLOADER_BASE_PATH = "sls://proj-a/logstore-a/genai"
_ITEM_URL = "sls://proj-a/logstore-a/genai/20260902/img.jpg"


class _RecordingRecorder:
    def __init__(self) -> None:
        self.successes: List[Tuple[str, int]] = []
        self.errors: List[Tuple[str, str]] = []
        self.terminal = threading.Event()

    def record_upload_success(
        self, *, provider: str, content_bytes: int
    ) -> None:
        self.successes.append((provider, content_bytes))
        self.terminal.set()

    def record_upload_error(self, *, provider: str, reason: str) -> None:
        self.errors.append((provider, reason))
        self.terminal.set()


class _FakePresignClient:
    """Presign client stub that records calls and replays canned results."""

    def __init__(self, results: Optional[List[Any]] = None) -> None:
        self.results = list(results or [])
        self.object_names: List[str] = []
        self.closed = False

    def presign(self, object_name: str):
        self.object_names.append(object_name)
        result = (
            self.results.pop(0)
            if self.results
            else _presigned(url=_SIGNED_URL)
        )
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        self.closed = True


def _presigned(**overrides: Any):
    from opentelemetry.util.genai._multimodal_upload.presign_client import (  # noqa: PLC0415
        PresignedUpload,
    )

    fields: Dict[str, Any] = {"url": _SIGNED_URL}
    fields.update(overrides)
    return PresignedUpload(**fields)


# Bound eagerly so helpers keep working while tests patch ``httpx.Client``.
_HTTPX_CLIENT_CLS = httpx.Client


def _mock_http_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return _HTTPX_CLIENT_CLS(transport=httpx.MockTransport(handler))


def _uploader(
    client: Any,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    base_path: str = _UPLOADER_BASE_PATH,
    **kwargs: Any,
) -> PresignUploader:
    return PresignUploader(
        base_path,
        client=client,
        max_workers=1,
        upload_retry_delay=0.0,
        http_client=_mock_http_client(handler),
        **kwargs,
    )


def _item(
    url: str = _ITEM_URL,
    **overrides: Any,
) -> UploadItem:
    fields: Dict[str, Any] = {
        "url": url,
        "content_type": "image/jpeg",
        "meta": {"traceId": "trace-1"},
        "data": b"jpeg-bytes",
    }
    fields.update(overrides)
    return UploadItem(**fields)


@pytest.fixture(name="recorder")
def _recorder_fixture():
    recorder = _RecordingRecorder()
    set_multimodal_usage_recorder(recorder)
    yield recorder
    set_multimodal_usage_recorder(None)


# --------------------------------------------------------------------------
# endpoint + response parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint", ["http://example.com", "https://example.com/base"]
)
@pytest.mark.parametrize("tls", ["true", "false"])
def test_endpoint_is_used_verbatim(monkeypatch, endpoint, tls) -> None:
    monkeypatch.setenv("APSARA_APM_COLLECTOR_USE_TLS", tls)
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, text=_SIGNED_URL)

    client = _presign_client(handler, endpoint=endpoint)
    try:
        client.presign("a.jpg")
        assert seen == [f"{endpoint}{PRESIGN_API_PATH}"]
    finally:
        client.close()


def test_endpoint_prefers_configured_value_over_arms_state() -> None:
    snapshot = replace(
        get_multimodal_config_snapshot(), presign_endpoint=_ENDPOINT
    )
    assert resolve_presign_endpoint(snapshot) == _ENDPOINT


def test_endpoint_falls_back_to_arms_state() -> None:
    snapshot = replace(get_multimodal_config_snapshot(), presign_endpoint=None)
    resolved = resolve_presign_endpoint(snapshot)
    assert resolved is None or isinstance(resolved, str)


@pytest.mark.parametrize(
    "payload",
    [
        {"uploadUrl": _SIGNED_URL},
        {"data": {"signedUrl": _SIGNED_URL}},
        {"result": {"data": {"url": _SIGNED_URL}}},
    ],
)
def test_presign_response_url_variants(payload: Any) -> None:
    assert parse_presign_response(payload).url == _SIGNED_URL


def test_presign_response_carries_method_headers_and_expiration() -> None:
    target = parse_presign_response(
        {
            "data": {
                "uploadUrl": _SIGNED_URL,
                "method": "post",
                "headers": {"Content-Type": "image/jpeg", "x-drop": None},
                "expireTime": 900,
            }
        }
    )
    assert target.method == "POST"
    assert target.headers == {"Content-Type": "image/jpeg"}
    assert target.expiration == "900"


@pytest.mark.parametrize(
    "payload",
    [
        {"success": False, "message": "workspace not authorized"},
        {"data": {}},
        [],
    ],
)
def test_unusable_presign_response_raises(payload: Any) -> None:
    with pytest.raises(PresignError):
        parse_presign_response(payload)


# --------------------------------------------------------------------------
# presign client
# --------------------------------------------------------------------------


def _presign_client(
    handler: Callable[[httpx.Request], httpx.Response],
    **overrides: Any,
) -> MultimodalPresignClient:
    kwargs: Dict[str, Any] = {
        "license_key": "test-license-key",
        "workspace": "test-workspace",
        "project": _PROJECT,
        "logstore": _LOGSTORE,
        "endpoint": _ENDPOINT,
    }
    kwargs.update(overrides)
    return MultimodalPresignClient(
        http_client=_mock_http_client(handler), **kwargs
    )


def test_presign_request_sends_license_key_workspace_and_sls_target() -> None:
    seen: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": {"uploadUrl": _SIGNED_URL}})

    client = _presign_client(handler)
    target = client.presign("genai/20260902/img.jpg")
    client.close()

    assert target.url == _SIGNED_URL
    assert seen["url"] == f"{_ENDPOINT}{PRESIGN_API_PATH}"
    assert seen["headers"][LICENSE_KEY_HEADER] == "test-license-key"
    assert seen["headers"][WORKSPACE_HEADER] == "test-workspace"
    assert seen["headers"]["content-type"] == "application/json"
    assert seen["body"] == {
        "objectName": "genai/20260902/img.jpg",
        "project": _PROJECT,
        "logstore": _LOGSTORE,
    }


def test_presign_client_requires_license_key() -> None:
    with pytest.raises(PresignConfigError):
        _presign_client(lambda request: httpx.Response(200), license_key=" ")


def test_presign_client_requires_endpoint_source() -> None:
    with pytest.raises(PresignConfigError):
        _presign_client(lambda request: httpx.Response(200), endpoint=None)


def test_presign_client_resolves_endpoint_lazily() -> None:
    endpoints = iter(["", _ENDPOINT])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"uploadUrl": _SIGNED_URL})

    client = MultimodalPresignClient(
        license_key="lk",
        endpoint_provider=lambda: next(endpoints, ""),
        http_client=_mock_http_client(handler),
    )
    with pytest.raises(PresignRetryableError):
        client.presign("a.jpg")
    assert client.presign("a.jpg").url == _SIGNED_URL
    client.close()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, PresignAuthError),
        (403, PresignAuthError),
        (429, PresignRetryableError),
        (503, PresignRetryableError),
        (400, PresignError),
    ],
)
def test_presign_error_classification(status: int, expected: type) -> None:
    client = _presign_client(lambda request: httpx.Response(status))
    with pytest.raises(expected):
        client.presign("a.jpg")
    client.close()


def test_presign_rejects_non_json_and_empty_object_name() -> None:
    client = _presign_client(
        lambda request: httpx.Response(200, text="not-json")
    )
    with pytest.raises(PresignError):
        client.presign("a.jpg")
    with pytest.raises(PresignConfigError):
        client.presign("/")
    client.close()


def test_presign_accepts_bare_url_response_body() -> None:
    """The ARMS API returns the presigned URL as a bare text body."""
    client = _presign_client(
        lambda request: httpx.Response(200, text=f"{_SIGNED_URL}\n")
    )
    target = client.presign("a.jpg")
    client.close()

    assert target.url == _SIGNED_URL
    assert target.method == "PUT"
    assert target.headers == {}


def test_presign_accepts_json_quoted_url_response_body() -> None:
    client = _presign_client(
        lambda request: httpx.Response(200, text=f'"{_SIGNED_URL}"')
    )
    target = client.presign("a.jpg")
    client.close()

    assert target.url == _SIGNED_URL


def test_presign_transport_error_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = _presign_client(handler)
    with pytest.raises(PresignRetryableError):
        client.presign("a.jpg")
    client.close()


# --------------------------------------------------------------------------
# uploader
# --------------------------------------------------------------------------


def test_upload_writes_object_to_presigned_url(recorder) -> None:
    seen: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["content_type"] = request.headers.get("content-type")
        seen["content"] = request.content
        return httpx.Response(200)

    client = _FakePresignClient()
    uploader = _uploader(client, handler)
    assert uploader.upload(_item())
    assert recorder.terminal.wait(2.0)
    uploader.shutdown(timeout=2.0)

    assert client.object_names == ["genai/20260902/img.jpg"]
    assert seen["method"] == "PUT"
    assert seen["url"] == _SIGNED_URL
    # OSS folds any header we add into the V4 canonical request, so the
    # uploader must not send an unsigned Content-Type.
    assert seen["content_type"] is None
    assert seen["content"] == b"jpeg-bytes"
    assert recorder.successes == [("oss", len(b"jpeg-bytes"))]
    assert not recorder.errors
    assert client.closed


def test_upload_honours_method_and_headers_from_presign_response() -> None:
    seen: Dict[str, Any] = {}
    done = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["content_type"] = request.headers.get("content-type")
        seen["extra"] = request.headers.get("x-oss-callback")
        done.set()
        return httpx.Response(200)

    client = _FakePresignClient(
        [
            _presigned(
                method="POST",
                headers={
                    "Content-Type": "application/octet-stream",
                    "x-oss-callback": "token",
                },
            )
        ]
    )
    uploader = _uploader(client, handler)
    assert uploader.upload(_item())
    assert done.wait(2.0)
    uploader.shutdown(timeout=2.0)

    assert seen["method"] == "POST"
    assert seen["content_type"] == "application/octet-stream"
    assert seen["extra"] == "token"


def test_upload_does_not_add_unsigned_content_type() -> None:
    """A self-added Content-Type makes OSS reject the presigned PUT (403)."""
    attempts: List[Optional[str]] = []
    done = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.headers.get("content-type"))
        done.set()
        return httpx.Response(200)

    client = _FakePresignClient()
    uploader = _uploader(client, handler)
    assert uploader.upload(_item())
    assert done.wait(2.0)
    uploader.shutdown(timeout=2.0)

    assert attempts == [None]


def test_upload_retries_transient_storage_failure(recorder) -> None:
    statuses = [503, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(statuses.pop(0))

    uploader = _uploader(_FakePresignClient(), handler)
    assert uploader.upload(_item())
    assert recorder.terminal.wait(2.0)
    uploader.shutdown(timeout=2.0)

    assert not statuses
    assert recorder.successes and not recorder.errors


def test_upload_does_not_retry_unauthorized_storage_response(
    recorder,
) -> None:
    calls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(
            403, text="<Error><Code>AccessDenied</Code></Error>"
        )

    uploader = _uploader(_FakePresignClient(), handler)
    assert uploader.upload(_item())
    assert recorder.terminal.wait(2.0)
    uploader.shutdown(timeout=2.0)

    assert len(calls) == 1
    assert recorder.errors == [("oss", "auth")]


def test_presign_auth_failure_skips_upload(recorder) -> None:
    calls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(200)

    client = _FakePresignClient([PresignAuthError("unauthorized")])
    uploader = _uploader(client, handler)
    assert uploader.upload(_item())
    assert recorder.terminal.wait(2.0)
    uploader.shutdown(timeout=2.0)

    assert not calls
    assert recorder.errors == [("oss", "auth")]


def test_presign_retryable_failure_is_retried_then_reported(recorder) -> None:
    client = _FakePresignClient(
        [
            PresignRetryableError("endpoint unavailable"),
            PresignRetryableError("endpoint unavailable"),
            PresignRetryableError("endpoint unavailable"),
        ]
    )
    uploader = _uploader(client, lambda request: httpx.Response(200))
    assert uploader.upload(_item())
    assert recorder.terminal.wait(2.0)
    uploader.shutdown(timeout=2.0)

    assert len(client.object_names) == 3
    assert recorder.errors == [("oss", "network")]


def test_upload_rejects_invalid_items(recorder) -> None:
    uploader = _uploader(
        _FakePresignClient(), lambda request: httpx.Response(200)
    )
    try:
        assert not uploader.upload(_item(data=None))
        assert not uploader.upload(
            _item(url="sls://other-project/logstore-a/genai/x.jpg")
        )
        assert not uploader.upload(
            _item(url="sls://proj-a/logstore-a/outside/x.jpg")
        )
    finally:
        uploader.shutdown(timeout=1.0)

    assert recorder.errors == [("oss", "invalid_item")] * 3


def test_upload_prefixes_relative_object_keys() -> None:
    done = threading.Event()
    client = _FakePresignClient()
    uploader = _uploader(
        client, lambda request: (done.set(), httpx.Response(200))[1]
    )
    assert uploader.upload(_item(url="20260902/img.jpg"))
    assert done.wait(2.0)
    uploader.shutdown(timeout=2.0)

    assert client.object_names == ["genai/20260902/img.jpg"]


def test_repeated_upload_of_same_object_is_skipped() -> None:
    done = threading.Event()
    client = _FakePresignClient()
    uploader = _uploader(
        client, lambda request: (done.set(), httpx.Response(200))[1]
    )
    item = _item()
    assert uploader.upload(item)
    assert done.wait(2.0)
    with uploader._queue_cond:  # pylint: disable=protected-access
        assert uploader._queue_cond.wait_for(  # pylint: disable=protected-access
            lambda: uploader._queue_count == 0,  # pylint: disable=protected-access
            timeout=2.0,
        )
    assert uploader.upload(item)
    uploader.shutdown(timeout=2.0)

    assert client.object_names == ["genai/20260902/img.jpg"]


def test_upload_downloads_source_uri_before_writing() -> None:
    done = threading.Event()
    seen: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["content"] = request.content
        done.set()
        return httpx.Response(200)

    uploader = _uploader(_FakePresignClient(), handler)
    uploader._download_content = lambda uri, *, max_size: b"downloaded"  # type: ignore[assignment]  # pylint: disable=protected-access
    assert uploader.upload(
        _item(
            data=None, source_uri="https://example.com/a.jpg", expected_size=9
        )
    )
    assert done.wait(2.0)
    uploader.shutdown(timeout=2.0)

    assert seen["content"] == b"downloaded"


@pytest.mark.parametrize(
    "base_path",
    [
        "file:///tmp",
        # oss:// is no longer an address the uploader understands: ARMS owns
        # the bucket, so objects are addressed by project/logstore.
        "oss://bucket-a/genai",
        "sls://proj-a",
        "sls://",
    ],
)
def test_invalid_base_path_is_rejected(base_path: str) -> None:
    with pytest.raises(PresignConfigError):
        _uploader(
            _FakePresignClient(),
            lambda request: httpx.Response(200),
            base_path=base_path,
        )


def test_upload_is_dropped_when_queue_is_full(recorder) -> None:
    release = threading.Event()
    running = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        running.set()
        assert release.wait(2.0)
        return httpx.Response(200)

    uploader = _uploader(
        _FakePresignClient(), handler, max_queue_size=1, max_queue_bytes=0
    )
    try:
        assert uploader.upload(_item(url=f"{_UPLOADER_BASE_PATH}/a.jpg"))
        assert running.wait(2.0)
        assert not uploader.upload(_item(url=f"{_UPLOADER_BASE_PATH}/b.jpg"))
        assert recorder.errors == [("oss", "queue_full")]
    finally:
        release.set()
        uploader.shutdown(timeout=2.0)


def test_upload_is_dropped_when_queue_bytes_exhausted(recorder) -> None:
    release = threading.Event()
    running = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        running.set()
        assert release.wait(2.0)
        return httpx.Response(200)

    uploader = _uploader(
        _FakePresignClient(), handler, max_queue_bytes=len(b"jpeg-bytes")
    )
    try:
        assert uploader.upload(_item(url=f"{_UPLOADER_BASE_PATH}/a.jpg"))
        assert running.wait(2.0)
        assert not uploader.upload(_item(url=f"{_UPLOADER_BASE_PATH}/b.jpg"))
        assert recorder.errors == [("oss", "queue_bytes_limit")]
    finally:
        release.set()
        uploader.shutdown(timeout=2.0)


def test_upload_after_shutdown_is_rejected(recorder) -> None:
    uploader = _uploader(
        _FakePresignClient(), lambda request: httpx.Response(200)
    )
    uploader.shutdown(timeout=1.0)
    uploader.shutdown(timeout=1.0)

    assert not uploader.upload(_item())
    assert recorder.errors == [("oss", "shutdown")]


@pytest.mark.parametrize("owns_http_client", [True, False])
@pytest.mark.parametrize("status", [200, 400])
def test_shutdown_timeout_closes_clients_after_last_task(
    monkeypatch, owns_http_client, status
) -> None:
    started = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    closed = threading.Event()
    requests = []
    close_calls = []
    presign_client = _FakePresignClient()

    def handler(request):
        index = len(requests)
        requests.append(request)
        started[index].set()
        assert release[index].wait(5.0)
        return httpx.Response(status)

    http_client = _mock_http_client(handler)
    original_close = http_client.close

    def close_http():
        close_calls.append("http")
        original_close()

    def close_presign():
        close_calls.append("presign")
        closed.set()

    monkeypatch.setattr(http_client, "close", close_http)
    monkeypatch.setattr(presign_client, "close", close_presign)
    monkeypatch.setattr(
        PresignUploader, "_new_http_client", lambda self: http_client
    )
    uploader = PresignUploader(
        _UPLOADER_BASE_PATH,
        client=presign_client,
        http_client=None if owns_http_client else http_client,
        max_workers=1,
        max_upload_attempts=1,
    )
    try:
        assert uploader.upload(_item())
        assert uploader.upload(_item(url=_ITEM_URL + ".second"))
        assert started[0].wait(2.0)
        uploader.shutdown(timeout=0.0)
        uploader.shutdown(timeout=0.0)
        assert not close_calls
        assert not http_client.is_closed

        release[0].set()
        assert started[1].wait(2.0)
        assert not close_calls
        release[1].set()
        assert closed.wait(2.0)
        uploader.shutdown(timeout=0.0)
        assert close_calls == (
            ["http", "presign"] if owns_http_client else ["presign"]
        )
        assert http_client.is_closed is owns_http_client
        assert uploader._queue_count == 0  # pylint: disable=protected-access
        assert uploader._current_queue_bytes == 0  # pylint: disable=protected-access
        assert not uploader._pending_paths  # pylint: disable=protected-access
    finally:
        for event in release:
            event.set()
        uploader.shutdown(timeout=2.0)
        uploader._executor.shutdown(wait=True)  # pylint: disable=protected-access
        original_close()


def test_downloads_reuse_upload_client(monkeypatch) -> None:
    requests = []

    def handler(request):
        requests.append((request.method, str(request.url)))
        return httpx.Response(200, content=b"image")

    uploader = _uploader(_FakePresignClient(), handler)

    def unexpected_client(*args, **kwargs):
        pytest.fail("Downloads must reuse the existing HTTP client")

    monkeypatch.setattr(
        "opentelemetry.util.genai._multimodal_upload.presign_uploader.httpx.Client",
        unexpected_client,
    )
    try:
        for index in range(2):
            assert uploader.upload(
                _item(
                    url=_ITEM_URL + str(index),
                    data=None,
                    source_uri=f"https://example.com/{index}.jpg",
                )
            )
        uploader.shutdown(timeout=2.0)
        assert [method for method, _ in requests] == [
            "GET",
            "PUT",
            "GET",
            "PUT",
        ]
        assert not uploader._http_client.is_closed  # pylint: disable=protected-access
    finally:
        uploader.shutdown(timeout=2.0)
        uploader._http_client.close()  # pylint: disable=protected-access


def test_executor_rejection_releases_queue_slot(recorder, monkeypatch) -> None:
    uploader = _uploader(
        _FakePresignClient(), lambda request: httpx.Response(200)
    )
    monkeypatch.setattr(
        uploader._executor,  # pylint: disable=protected-access
        "submit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("shutting down")
        ),
    )
    try:
        assert not uploader.upload(_item())
        assert uploader._queue_count == 0  # pylint: disable=protected-access
        assert not uploader._pending_paths  # pylint: disable=protected-access
        assert recorder.errors == [("oss", "shutdown")]
    finally:
        uploader.shutdown(timeout=1.0)


def test_fork_reinit_rebuilds_worker_state() -> None:
    done = threading.Event()
    uploader = _uploader(
        _FakePresignClient(),
        lambda request: (done.set(), httpx.Response(200))[1],
    )
    uploader._at_fork_reinit()  # pylint: disable=protected-access
    assert uploader.upload(_item())
    assert done.wait(2.0)
    uploader.shutdown(timeout=2.0)


def test_download_source_content_enforces_size_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"0123456789")

    uploader = _uploader(_FakePresignClient(), handler)
    try:
        assert (
            uploader._download_content(  # pylint: disable=protected-access
                "https://example.com/a.jpg", max_size=64
            )
            == b"0123456789"
        )
        assert (
            uploader._download_content(  # pylint: disable=protected-access
                "https://example.com/a.jpg", max_size=4
            )
            is None
        )
    finally:
        uploader.shutdown(timeout=1.0)


@pytest.mark.parametrize("status", [302, 404])
def test_download_source_content_rejects_bad_responses(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers={"Location": "https://other"})

    uploader = _uploader(_FakePresignClient(), handler)
    try:
        assert (
            uploader._download_content(  # pylint: disable=protected-access
                "https://example.com/a.jpg", max_size=64
            )
            is None
        )
    finally:
        uploader.shutdown(timeout=1.0)


def test_failed_download_reports_error(recorder) -> None:
    uploader = _uploader(
        _FakePresignClient(), lambda request: httpx.Response(200)
    )
    uploader._download_content = lambda uri, *, max_size: None  # type: ignore[assignment]  # pylint: disable=protected-access
    assert uploader.upload(
        _item(data=None, source_uri="https://example.com/a.jpg")
    )
    assert recorder.terminal.wait(2.0)
    uploader.shutdown(timeout=2.0)

    assert recorder.errors == [("oss", "download_failed")]


# --------------------------------------------------------------------------
# hooks + runtime config
# --------------------------------------------------------------------------


def _presign_env(**overrides: str) -> Dict[str, str]:
    env = {
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: PRESIGN_HOOK_NAME,
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: PRESIGN_HOOK_NAME,
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_ENDPOINT: _ENDPOINT,
        _SLS_PROJECT_ENV: _PROJECT,
        _SLS_LOGSTORE_ENV: _LOGSTORE,
        "ARMS_LICENSE_KEY": "test-license-key",
        "ARMS_WORKSPACE": "test-workspace",
    }
    env.update(overrides)
    return env


def _apply_env(monkeypatch, env: Dict[str, str]) -> None:
    for key in set(_presign_env()) | {
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_BUCKET,
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_PATH_PREFIX,
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_LICENSE_KEY,
        OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_WORKSPACE,
    }:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    reset_multimodal_runtime_state_for_test()


def test_hooks_build_presign_pair_from_environment(monkeypatch) -> None:
    _apply_env(monkeypatch, _presign_env())
    snapshot = get_multimodal_config_snapshot()
    assert snapshot.uploader_hook_name == PRESIGN_HOOK_NAME
    assert snapshot.effective_storage_base_path == _BASE_PATH

    uploader = presign_uploader_hook()
    pre_uploader = presign_pre_uploader_hook()
    assert isinstance(uploader, PresignUploader)
    assert uploader.base_path == _BASE_PATH
    assert uploader.project == _PROJECT
    assert uploader.logstore == _LOGSTORE
    assert pre_uploader is not None
    assert pre_uploader.base_path == _BASE_PATH
    uploader.shutdown(timeout=1.0)
    pre_uploader.shutdown(timeout=1.0)


def test_hooks_apply_configured_path_prefix(monkeypatch) -> None:
    _apply_env(
        monkeypatch,
        _presign_env(
            **{
                OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_PATH_PREFIX: (
                    "/genai/app-a/"
                )
            }
        ),
    )
    uploader = presign_uploader_hook()
    assert isinstance(uploader, PresignUploader)
    assert uploader.base_path == f"{_BASE_PATH}/genai/app-a"
    uploader.shutdown(timeout=1.0)


def test_oss_bucket_is_ignored_in_presign_mode(monkeypatch) -> None:
    """OSS_BUCKET is deprecated here: ARMS decides which bucket backs objects."""
    _apply_env(
        monkeypatch,
        _presign_env(
            **{OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_OSS_BUCKET: "bucket-a"}
        ),
    )
    snapshot = get_multimodal_config_snapshot()
    assert snapshot.oss_bucket == "bucket-a"
    assert snapshot.effective_storage_base_path == _BASE_PATH


@pytest.mark.parametrize(
    "missing",
    [
        _SLS_PROJECT_ENV,
        "ARMS_LICENSE_KEY",
    ],
)
def test_hooks_disable_pair_when_config_incomplete(
    monkeypatch, missing: str
) -> None:
    env = _presign_env()
    env.pop(missing)
    _apply_env(monkeypatch, env)
    if missing == _SLS_PROJECT_ENV:
        # A co-located ARMS agent may still expose a project, so force the
        # "no project anywhere" case to keep the degradation deterministic.
        monkeypatch.setattr(
            "opentelemetry.util.genai._multimodal_upload.presign_uploader"
            "._resolve_sls_target",
            lambda snapshot: ("", DEFAULT_SLS_LOGSTORE),
        )

    assert presign_uploader_hook() is None
    assert presign_pre_uploader_hook() is None


def test_pre_uploader_hook_requires_resolvable_endpoint(monkeypatch) -> None:
    env = _presign_env()
    env.pop(OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_ENDPOINT)
    _apply_env(monkeypatch, env)
    monkeypatch.setattr(
        "opentelemetry.util.genai._multimodal_upload.presign_uploader"
        ".resolve_presign_endpoint",
        lambda snapshot: None,
    )

    assert presign_pre_uploader_hook() is None
    assert presign_uploader_hook() is None


@pytest.mark.parametrize(
    "raw",
    ["presign", "PRESIGN", "oss-presign", "oss_presign", "presigned-oss"],
)
def test_console_hook_name_aliases(raw: str) -> None:
    assert normalize_multimodal_hook_name(raw) == PRESIGN_HOOK_NAME


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("bucket-a", "bucket-a"),
        (" oss://bucket-a/genai/ ", "bucket-a/genai"),
        ("", None),
        (None, None),
        ("s3://bucket-a", None),
        ("oss://bucket-a/../etc", None),
    ],
)
def test_oss_bucket_normalization(raw: Any, expected: Optional[str]) -> None:
    assert normalize_oss_bucket(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("genai", "genai"),
        (" /genai/app-a/ ", "genai/app-a"),
        ("", None),
        (None, None),
        ("genai/../etc", None),
        ("genai//app-a", None),
    ],
)
def test_oss_path_prefix_normalization(
    raw: Any, expected: Optional[str]
) -> None:
    assert normalize_oss_path_prefix(raw) == expected


def test_path_prefix_change_bumps_uploader_generation() -> None:
    before = update_multimodal_runtime_config(
        upload_mode="both",
        uploader_hook_name=PRESIGN_HOOK_NAME,
        pre_uploader_hook_name=PRESIGN_HOOK_NAME,
        sls_project=_PROJECT,
        sls_logstore=_LOGSTORE,
        oss_path_prefix="genai",
    )
    assert before.effective_storage_base_path == f"{_BASE_PATH}/genai"
    assert "oss_path_prefix" in UPLOADER_GENERATION_FIELDS

    after = update_multimodal_runtime_config(oss_path_prefix="/genai/app-b/")
    assert after.oss_path_prefix == "genai/app-b"
    assert after.effective_storage_base_path == f"{_BASE_PATH}/genai/app-b"
    assert after.uploader_generation > before.uploader_generation


def test_presign_client_payload_uses_snapshot_sls_target(monkeypatch) -> None:
    _apply_env(monkeypatch, _presign_env())
    snapshot = replace(
        get_multimodal_config_snapshot(),
        sls_project="proj-xtrace-test",
        sls_logstore="logstore-multimodal",
    )
    client = build_presign_client(snapshot)
    try:
        # No bucket field: ARMS resolves the backing bucket itself.
        assert client.build_payload("genai/img.jpg") == {
            "objectName": "genai/img.jpg",
            "project": "proj-xtrace-test",
            "logstore": "logstore-multimodal",
        }
    finally:
        client.close()


def test_sls_target_falls_back_to_arms_state() -> None:
    snapshot = replace(
        get_multimodal_config_snapshot(), sls_project=None, sls_logstore=None
    )
    project, logstore = _resolve_sls_target(snapshot)
    assert isinstance(project, str)
    # The logstore always defaults so the recorded URI and the presign request
    # agree on where the object lands.
    assert logstore == DEFAULT_SLS_LOGSTORE


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 30.0), ("5.5", 5.5), ("0", 30.0), ("abc", 30.0)],
)
def test_presign_timeout_env_parsing(
    monkeypatch, raw: Optional[str], expected: float
) -> None:
    if raw is None:
        monkeypatch.delenv(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_TIMEOUT,
            raising=False,
        )
    else:
        monkeypatch.setenv(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_TIMEOUT, raw
        )
    assert _presign_timeout() == expected


def test_entry_point_loading_builds_presign_pair(monkeypatch) -> None:
    _apply_env(monkeypatch, _presign_env())
    from opentelemetry.util.genai._multimodal_upload import (  # noqa: PLC0415
        multimodal_upload_hook,
    )
    from opentelemetry.util.genai._multimodal_upload.pre_uploader import (  # noqa: PLC0415
        MultimodalPreUploader,
    )

    hooks = {
        "opentelemetry_genai_multimodal_uploader": presign_uploader_hook,
        "opentelemetry_genai_multimodal_pre_uploader": presign_pre_uploader_hook,
    }

    class _EntryPoint:
        def __init__(self, hook: Callable[..., Any]) -> None:
            self.name = PRESIGN_HOOK_NAME
            self._hook = hook

        def load(self) -> Callable[..., Any]:
            return self._hook

    monkeypatch.setattr(
        multimodal_upload_hook,
        "_iter_entry_points",
        lambda group: [_EntryPoint(hooks[group])] if group in hooks else [],
    )

    uploader, pre_uploader = (
        multimodal_upload_hook.get_or_rebuild_uploader_pair()
    )
    try:
        assert isinstance(uploader, PresignUploader)
        assert isinstance(pre_uploader, MultimodalPreUploader)
        assert uploader.base_path == pre_uploader.base_path == _BASE_PATH
    finally:
        if uploader is not None:
            uploader.shutdown(timeout=1.0)
        if pre_uploader is not None:
            pre_uploader.shutdown(timeout=1.0)


def test_presign_credentials_are_not_runtime_updatable(monkeypatch) -> None:
    _apply_env(monkeypatch, _presign_env())
    before = get_multimodal_config_snapshot()
    after = update_multimodal_runtime_config(
        presign_license_key="rotated",
        presign_workspace="rotated",
        presign_endpoint="rotated",
    )
    assert after == before


def test_presign_identity_can_come_from_dedicated_env(monkeypatch) -> None:
    """The identity is configurable without an ARMS agent in the process."""
    env = _presign_env()
    env.pop("ARMS_LICENSE_KEY")
    env.pop("ARMS_WORKSPACE")
    env[OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_LICENSE_KEY] = "own-key"
    env[OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_WORKSPACE] = "own-ws"
    _apply_env(monkeypatch, env)

    snapshot = get_multimodal_config_snapshot()
    assert snapshot.presign_license_key == "own-key"
    assert snapshot.presign_workspace == "own-ws"


def test_dedicated_presign_identity_wins_over_arms_env(monkeypatch) -> None:
    """ARMS_* stays a fallback, so an explicit setting must take precedence."""
    _apply_env(
        monkeypatch,
        _presign_env(
            **{
                OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_LICENSE_KEY: "own-key",
                OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRESIGN_WORKSPACE: "own-ws",
            }
        ),
    )
    snapshot = get_multimodal_config_snapshot()
    assert snapshot.presign_license_key == "own-key"
    assert snapshot.presign_workspace == "own-ws"


def test_presign_identity_falls_back_to_arms_env(monkeypatch) -> None:
    """A co-located ARMS agent keeps working without extra configuration."""
    _apply_env(monkeypatch, _presign_env())
    snapshot = get_multimodal_config_snapshot()
    assert snapshot.presign_license_key == "test-license-key"
    assert snapshot.presign_workspace == "test-workspace"


@pytest.mark.parametrize("expected_size", [0, 1])
@pytest.mark.parametrize("content", [b"1234", b"123456789"])
def test_download_reserves_budget_and_enforces_stream_limit(
    recorder, expected_size, content
) -> None:
    started = threading.Event()
    release = threading.Event()
    uploading = threading.Event()
    finish_upload = threading.Event()
    client = _FakePresignClient()

    def handler(request):
        if request.method == "GET":
            started.set()
            assert release.wait(5.0)
            return httpx.Response(200, content=content)
        uploading.set()
        assert finish_upload.wait(5.0)
        return httpx.Response(200)

    uploader = _uploader(client, handler, max_queue_bytes=8)
    item = _item(
        data=None,
        source_uri="https://example.com/a",
        expected_size=expected_size,
    )
    try:
        assert uploader.upload(item)
        assert started.wait(2.0)
        assert uploader._current_queue_bytes == 8  # pylint: disable=protected-access
        assert not uploader.upload(_item(url=_ITEM_URL + ".second", data=b"x"))
        assert recorder.errors == [("oss", "queue_bytes_limit")]
        release.set()
        if len(content) <= 8:
            assert uploading.wait(2.0)
            assert uploader._current_queue_bytes == 4  # pylint: disable=protected-access
            # The unused reservation becomes available before upload completes.
            assert uploader.upload(
                _item(url=_ITEM_URL + ".third", data=b"1234")
            )
        finish_upload.set()
        uploader.shutdown(timeout=2.0)
        if len(content) > 8:
            assert not client.object_names
            assert recorder.errors[-1] == ("oss", "download_failed")
        else:
            assert len(client.object_names) == 2
        assert uploader._queue_count == 0  # pylint: disable=protected-access
        assert uploader._current_queue_bytes == 0  # pylint: disable=protected-access
        assert not uploader._pending_paths  # pylint: disable=protected-access
    finally:
        release.set()
        finish_upload.set()
        uploader.shutdown(timeout=2.0)
        uploader._executor.shutdown(wait=True)  # pylint: disable=protected-access
        uploader._http_client.close()  # pylint: disable=protected-access


def test_download_rejects_content_exceeding_reservation(recorder) -> None:
    client = _FakePresignClient()
    uploader = _uploader(
        client, lambda request: httpx.Response(200), max_queue_bytes=4
    )
    uploader._download_content = lambda uri, *, max_size: b"12345"  # pylint: disable=protected-access
    try:
        assert uploader.upload(
            _item(
                data=None, source_uri="https://example.com/a", expected_size=0
            )
        )
        uploader.shutdown(timeout=2.0)
        assert recorder.errors == [("oss", "queue_bytes_limit")]
        assert not client.object_names
        assert uploader._current_queue_bytes == 0  # pylint: disable=protected-access
        assert uploader._queue_count == 0  # pylint: disable=protected-access
        assert not uploader._pending_paths  # pylint: disable=protected-access
    finally:
        uploader.shutdown(timeout=2.0)
        uploader._http_client.close()  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "expiration, expected",
    [(0, "0"), (None, None), ("", None), ("  ", None), (" 900 ", "900")],
)
def test_presign_expiration_preserves_zero_and_trims(
    expiration, expected
) -> None:
    target = parse_presign_response(
        {"url": _SIGNED_URL, "expiration": expiration}
    )
    assert target.expiration == expected
    if expected is None:
        target = parse_presign_response(
            {"url": _SIGNED_URL, "expiration": expiration, "expireTime": 0}
        )
        assert target.expiration == "0"


def test_sls_target_retains_default_and_prefix() -> None:
    assert (
        format_sls_base_path("project", None, "a/b")
        == f"sls://project/{DEFAULT_SLS_LOGSTORE}/a/b"
    )
    assert (
        format_sls_base_path("project", "logstore", "a/b")
        == "sls://project/logstore/a/b"
    )


def test_build_client_follows_endpoint_changes_after_readiness_check(
    monkeypatch,
) -> None:
    _apply_env(monkeypatch, _presign_env())
    endpoints = iter(
        [
            "https://initial.example",
            "https://next.example",
            "https://final.example",
        ]
    )
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, text=_SIGNED_URL)

    monkeypatch.setattr(
        "opentelemetry.util.genai._multimodal_upload.presign_uploader.resolve_presign_endpoint",
        lambda snapshot: next(endpoints),
    )
    monkeypatch.setattr(
        "opentelemetry.util.genai._multimodal_upload.presign_client.httpx.Client",
        lambda **kwargs: _mock_http_client(handler),
    )
    client = build_presign_client(get_multimodal_config_snapshot())
    try:
        client.presign("a.jpg")
        client.presign("b.jpg")
        assert seen == [
            f"https://next.example{PRESIGN_API_PATH}",
            f"https://final.example{PRESIGN_API_PATH}",
        ]
    finally:
        client.close()
