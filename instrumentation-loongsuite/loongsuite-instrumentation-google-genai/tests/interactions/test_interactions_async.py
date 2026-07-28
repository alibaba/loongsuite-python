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
