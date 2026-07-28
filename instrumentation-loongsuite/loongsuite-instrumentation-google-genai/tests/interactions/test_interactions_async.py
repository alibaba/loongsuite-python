# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

from .base import TestCase


class TestInteractionsAsync(TestCase):
    def run_interaction(self, *args: Any, **kwargs: Any) -> Any:
        return asyncio.run(
            self.client.aio.interactions.create(*args, **kwargs)
        )

    def run_streaming_interaction(
        self, *args: Any, **kwargs: Any
    ) -> list[Any]:
        async def _run() -> list[Any]:
            stream = await self.client.aio.interactions.create(*args, **kwargs)
            events = []
            async for event in stream:
                events.append(event)
            return events

        return asyncio.run(_run())
