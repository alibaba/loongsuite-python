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

"""A2A protocol tracing following the experimental OTel A2A proposal.

Agent, model and tool spans belong to their framework instrumentations.
"""

import importlib
import os
from contextlib import nullcontext
from typing import Collection

from wrapt import FunctionWrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import (
    INVALID_SPAN,
    NoOpTracer,
    NoOpTracerProvider,
    get_tracer,
)

from ._semconv import CLIENT_METHODS, SERVER_METHODS, STREAM_METHODS
from ._wrappers import wrap
from .package import _instruments
from .version import __version__


class _SDKNoOpTracer(NoOpTracer):
    def start_as_current_span(self, *args, **kwargs):
        # SDK decorators must neither replace the transport's current span nor
        # record their internal status/errors onto it while tracing is disabled.
        return nullcontext(INVALID_SPAN)


class _SDKNoOpTracerProvider(NoOpTracerProvider):
    def get_tracer(self, *args, **kwargs):
        return _SDKNoOpTracer()


class A2AInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        self._tracer = get_tracer(
            __name__, __version__, kwargs.get("tracer_provider")
        )
        self._patched = []
        self._native = None
        # The SDK resolves its module-local tracer at call time, including decorators
        # installed before instrument(). Do not mutate the global tracer or environment.
        telemetry = importlib.import_module("a2a.utils.telemetry")
        if (
            os.getenv("OTEL_INSTRUMENTATION_A2A_SDK_ENABLED", "false").lower()
            != "true"
        ):
            self._native = (telemetry, telemetry.trace)
            telemetry.trace = _SDKNoOpTracerProvider()

        from a2a.server.request_handlers import RequestHandler

        self._watch(RequestHandler, SERVER_METHODS, True)
        # HTTP transports are optional in older SDK versions. No gRPC imports or
        # dependencies are required for HTTP-only applications.
        for module_name, class_names in (
            (
                "a2a.client.transports.jsonrpc",
                ("JsonRpcTransport", "JsonRpcClient"),
            ),
            ("a2a.client.transports.rest", ("RestTransport", "RestClient")),
            ("a2a.client", ("A2AClient",)),
        ):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            for class_name in class_names:
                cls = getattr(module, class_name, None)
                if cls is not None:
                    self._patch_class(cls, CLIENT_METHODS, False)
        try:
            rpc = importlib.import_module(
                "a2a.server.request_handlers.jsonrpc_handler"
            )
        except ImportError:
            pass
        else:
            self._patch_class(rpc.JSONRPCHandler, SERVER_METHODS, True)

    def _patch_class(self, cls, methods, server):
        for name, method in methods.items():
            original = cls.__dict__.get(name)
            if original is None or any(
                c is cls and n == name for c, n, _, _ in self._patched
            ):
                continue
            wrapper = FunctionWrapper(
                original,
                wrap(self._tracer, method, server, method in STREAM_METHODS),
            )
            setattr(cls, name, wrapper)
            self._patched.append((cls, name, original, wrapper))

    def _watch(self, base, methods, server):
        def visit(cls):
            self._patch_class(cls, methods, server)
            for child in cls.__subclasses__():
                visit(child)

        visit(base)
        own = base.__dict__.get("__init_subclass__")

        def init_subclass(cls, **kwargs):
            if own is not None:
                own.__get__(None, cls)(**kwargs)
            else:
                super(base, cls).__init_subclass__(**kwargs)
            self._patch_class(cls, methods, server)

        replacement = classmethod(init_subclass)
        setattr(base, "__init_subclass__", replacement)
        self._subclass_hook = (base, own, replacement)

    def _uninstrument(self, **kwargs):
        for cls, name, original, wrapper in reversed(self._patched):
            if cls.__dict__.get(name) is wrapper:
                setattr(cls, name, original)
        self._patched.clear()
        base, own, replacement = self._subclass_hook
        if base.__dict__.get("__init_subclass__") is replacement:
            if own is None:
                delattr(base, "__init_subclass__")
            else:
                setattr(base, "__init_subclass__", own)
        if self._native is not None:
            module, original = self._native
            module.trace = original
            self._native = None
