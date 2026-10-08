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

"""Probe-only operations. Protocol spans never serialize message content."""

from dataclasses import dataclass

from opentelemetry import context, propagate, trace
from opentelemetry.instrumentation.utils import is_instrumentation_enabled
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.genai import hook_advice

from ._semconv import attributes, card_attributes, field, payload, terminal

_CLIENT = context.create_key("a2a-client")
_SERVER = context.create_key("a2a-server")
_HEADERS = context.create_key("a2a-headers")


@dataclass
class Operation:
    span: object
    parent: object
    owned: bool
    key: str
    ended: bool = False


@hook_advice("a2a", "start")
def start(tracer, method, instance, args, kwargs, server):
    key = _SERVER if server else _CLIENT
    if not is_instrumentation_enabled() or context.get_value(key):
        return None
    request = args[0] if args else kwargs.get("request", kwargs.get("params"))
    attrs = {
        "a2a.method.name": method,
        **card_attributes(instance),
        **attributes(request, request=True),
    }
    parent = context.get_current()
    current = trace.get_current_span()
    # Only borrow a detectable transport SERVER span, never an arbitrary ancestor.
    current_attrs = getattr(current, "attributes", {}) or {}
    borrowed = (
        server
        and getattr(current, "kind", None) == SpanKind.SERVER
        and any(
            key in current_attrs
            for key in (
                "http.request.method",
                "http.method",
                "rpc.system",
                "rpc.system.name",
            )
        )
    )
    if server:
        call_context = kwargs.get("context") or (
            args[1] if len(args) > 1 else None
        )
        headers = field(call_context, "state", {}).get(
            "headers", context.get_value(_HEADERS) or {}
        )
        if not borrowed:
            # Incoming requests are remote boundaries, not children of incidental local context.
            parent = propagate.extract(headers, context=context.Context())
        tenant = field(call_context, "tenant")
        if tenant:
            attrs["a2a.tenant"] = tenant
        version = headers.get("a2a-version")
        if version:
            attrs["a2a.protocol.version"] = version
    span = (
        current
        if borrowed
        else tracer.start_span(
            method,
            kind=SpanKind.SERVER if server else SpanKind.CLIENT,
            context=parent,
            attributes=attrs,
        )
    )
    state = Operation(span, parent, not borrowed, key)
    try:
        if borrowed:
            span.set_attributes(attrs)
            if (
                "http.request.method" in current_attrs
                or "http.method" in current_attrs
            ):
                span.update_name(method)
        return state
    except Exception:
        finish(state)
        raise


@hook_advice("a2a", "attach")
def attach(state):
    if state is None:
        return None
    ctx = trace.set_span_in_context(state.span, state.parent)
    ctx = context.set_value(state.key, True, ctx)
    return context.attach(ctx)


@hook_advice("a2a", "detach")
def detach(token):
    if token is not None:
        context.detach(token)


@hook_advice("a2a", "response")
def response(state, value):
    if state is None or state.ended:
        return
    state.span.set_attributes(attributes(value))
    error_value = field(payload(value), "error")
    if error_value is not None:
        state.span.set_attribute(
            "error.type", str(field(error_value, "code", "A2AError"))
        )
        state.span.set_status(StatusCode.ERROR)


@hook_advice("a2a", "error")
def error(state, exception):
    if state is not None and not state.ended:
        state.span.set_attribute("error.type", type(exception).__qualname__)
        state.span.set_status(StatusCode.ERROR)
        # Exception messages can contain request content. Do not capture them.


@hook_advice("a2a", "finish")
def finish(state):
    if state is not None and not state.ended:
        state.ended = True
        if state.owned:
            state.span.end()


@hook_advice("a2a", "terminal")
def is_terminal(value):
    return terminal(value)


@hook_advice("a2a", "inject")
def inject(request):
    if context.get_value(_CLIENT) and is_instrumentation_enabled():
        # The HTTPX transport probe, when installed, injects its own child context later.
        carrier = {}
        propagate.inject(carrier)
        request.headers.update(carrier)


@hook_advice("a2a", "request-headers")
def request_headers(request):
    return context.attach(context.set_value(_HEADERS, dict(request.headers)))
