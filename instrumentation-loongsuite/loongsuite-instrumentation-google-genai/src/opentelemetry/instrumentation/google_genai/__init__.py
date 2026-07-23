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

"""LoongSuite instrumentation for the Google GenAI Python SDK."""

from __future__ import annotations

import logging
from typing import Any, Collection

from google.genai.models import AsyncModels, Models

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

from ._context import GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY
from ._wrappers import (
    create_async_embedding_wrapper,
    create_async_generate_wrapper,
    create_sync_embedding_wrapper,
    create_sync_generate_wrapper,
)
from .package import _instruments
from .version import __version__

_logger = logging.getLogger(__name__)


class GoogleGenAiSdkInstrumentor(BaseInstrumentor):
    """Instrument Google GenAI generation, streaming, and embedding calls."""

    def __init__(self):
        super().__init__()
        self._originals: dict[tuple[type, str], Any] = {}
        self._handler: ExtendedTelemetryHandler | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _save_and_replace(self, target: type, name: str, replacement) -> None:
        self._originals[(target, name)] = getattr(target, name)
        setattr(target, name, replacement)

    def _instrument(self, **kwargs: Any) -> None:
        self._handler = ExtendedTelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
        )
        try:
            sync_generate = Models.generate_content
            sync_stream = Models.generate_content_stream
            async_generate = AsyncModels.generate_content
            async_stream = AsyncModels.generate_content_stream
            sync_embedding = Models.embed_content
            async_embedding = AsyncModels.embed_content

            self._save_and_replace(
                Models,
                "generate_content",
                create_sync_generate_wrapper(
                    sync_generate, self._handler, streaming=False
                ),
            )
            self._save_and_replace(
                Models,
                "generate_content_stream",
                create_sync_generate_wrapper(
                    sync_stream, self._handler, streaming=True
                ),
            )
            self._save_and_replace(
                AsyncModels,
                "generate_content",
                create_async_generate_wrapper(
                    async_generate, self._handler, streaming=False
                ),
            )
            self._save_and_replace(
                AsyncModels,
                "generate_content_stream",
                create_async_generate_wrapper(
                    async_stream, self._handler, streaming=True
                ),
            )
            self._save_and_replace(
                Models,
                "embed_content",
                create_sync_embedding_wrapper(sync_embedding, self._handler),
            )
            self._save_and_replace(
                AsyncModels,
                "embed_content",
                create_async_embedding_wrapper(async_embedding, self._handler),
            )
        except Exception:
            self._restore_originals()
            self._handler = None
            raise

    def _restore_originals(self) -> None:
        for (target, name), original in reversed(
            list(self._originals.items())
        ):
            setattr(target, name, original)
        self._originals.clear()

    def _uninstrument(self, **kwargs: Any) -> None:
        self._restore_originals()
        self._handler = None


__all__ = [
    "GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY",
    "GoogleGenAiSdkInstrumentor",
    "__version__",
]
