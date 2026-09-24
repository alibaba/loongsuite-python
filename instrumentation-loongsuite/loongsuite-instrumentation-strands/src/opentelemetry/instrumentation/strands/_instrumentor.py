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

import logging
from collections.abc import AsyncGenerator
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import context as otel_context
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.strands._hooks import LoongsuiteHook
from opentelemetry.instrumentation.strands._native_telemetry import (
    NativeTelemetrySuppression,
)
from opentelemetry.instrumentation.strands.package import _instruments
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)


class StrandsInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        self._hook = LoongsuiteHook(handler)
        self._native_telemetry = NativeTelemetrySuppression()

        try:
            self._native_telemetry.install()
        except Exception:
            self._native_telemetry.restore()
            logger.warning(
                "Failed to suppress Strands native telemetry; "
                "LoongSuite instrumentation will continue",
                exc_info=True,
            )

        try:
            wrap_function_wrapper(
                module="strands.agent.agent",
                name="Agent.__init__",
                wrapper=self._agent_init_wrapper,
            )
            wrap_function_wrapper(
                module="strands.agent.agent",
                name="Agent.stream_async",
                wrapper=self._stream_async_wrapper,
            )
        except Exception:
            self._unwrap_agent_lifecycle()
            self._native_telemetry.restore()
            logger.warning(
                "Failed to wrap strands Agent lifecycle", exc_info=True
            )

    @staticmethod
    def _unwrap_agent_lifecycle() -> None:
        try:
            import strands.agent.agent as agent_module  # noqa: PLC0415

            unwrap(agent_module.Agent, "__init__")
            unwrap(agent_module.Agent, "stream_async")
        except Exception:
            pass

    def _agent_init_wrapper(self, wrapped, instance, args, kwargs):
        wrapped(*args, **kwargs)
        try:
            if getattr(instance, "_loongsuite_strands_hook", None) is None:
                instance.hooks.add_hook(self._hook)
                instance._loongsuite_strands_hook = self._hook
        except Exception:
            logger.debug(
                "Failed to register loongsuite hook on Agent", exc_info=True
            )

    def _stream_async_wrapper(self, wrapped, instance, args, kwargs):
        return self._forward_stream(wrapped(*args, **kwargs))

    async def _forward_stream(self, stream: Any) -> AsyncGenerator[Any, None]:
        state_key = None
        iterator = stream.__aiter__()
        try:
            while True:
                baseline_context = otel_context.get_current()
                self._safe_hook_call(
                    self._hook.attach_stream_contexts, state_key
                )
                try:
                    event = await iterator.__anext__()
                except StopAsyncIteration:
                    return
                finally:
                    state_key = (
                        state_key or self._hook.current_invocation_key()
                    )
                    self._safe_hook_call(
                        self._hook.detach_stream_contexts, state_key
                    )
                    if otel_context.get_current() is not baseline_context:
                        self._safe_hook_call(
                            self._hook.abandon_stream_context_tokens, state_key
                        )
                        otel_context.attach(baseline_context)
                yield event
        except BaseException as exc:
            self._safe_hook_call(self._hook.attach_stream_contexts, state_key)
            self._safe_hook_call(
                self._hook.finish_failed_stream, state_key, exc
            )
            raise
        finally:
            try:
                close = getattr(iterator, "aclose", None)
                if callable(close):
                    await close()
            except Exception:
                logger.debug(
                    "Failed to close strands stream during telemetry cleanup",
                    exc_info=True,
                )
            finally:
                self._safe_hook_call(
                    self._hook.attach_stream_contexts, state_key
                )
                self._safe_hook_call(
                    self._hook.finish_closed_stream, state_key
                )

    @staticmethod
    def _safe_hook_call(callback: Any, *args: Any) -> None:
        try:
            callback(*args)
        except Exception:
            logger.debug("Failed to run strands telemetry hook", exc_info=True)

    def _uninstrument(self, **kwargs):
        self._unwrap_agent_lifecycle()
        native_telemetry = getattr(self, "_native_telemetry", None)
        if native_telemetry is not None:
            native_telemetry.restore()
