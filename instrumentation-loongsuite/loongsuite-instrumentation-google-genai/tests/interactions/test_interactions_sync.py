# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from .base import TestCase


class TestInteractionsSync(TestCase):
    def run_interaction(self, *args: Any, **kwargs: Any) -> Any:
        return self.client.interactions.create(*args, **kwargs)

    def run_streaming_interaction(
        self, *args: Any, **kwargs: Any
    ) -> list[Any]:
        stream = self.client.interactions.create(*args, **kwargs)
        return list(stream)
