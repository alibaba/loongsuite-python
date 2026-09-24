#!/usr/bin/env python3

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

"""Local MCP server used by Hermes Agent instrumentation tests."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("LoongSuite Hermes Demo MCP")


def _knowledge_base() -> dict[str, object]:
    path = os.environ.get("HERMES_DEMO_KB_PATH")
    if not path:
        return {"briefings": []}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"briefings": []}


@mcp.tool()
def get_current_time(timezone: str = "UTC") -> dict[str, str]:
    """Return the current time in ISO-8601 format for the requested timezone."""

    try:
        tz = ZoneInfo(timezone)
    except Exception:  # noqa: BLE001 - keep the demo tool dependency-light.
        tz = ZoneInfo("UTC")
        timezone = "UTC"
    return {
        "timezone": timezone,
        "iso": datetime.now(tz).isoformat(),
    }


@mcp.tool()
def search_briefing(query: str) -> dict[str, str]:
    """Return a deterministic demo briefing answer for a query."""

    normalized_query = (query or "").strip().lower()
    for item in _knowledge_base().get("briefings", []):
        if not isinstance(item, dict):
            continue
        if item.get("query") == normalized_query:
            return {"answer": str(item.get("answer", ""))}
    return {"answer": "No briefing found."}


@mcp.tool()
def grade_candidate(
    reference: str,
    candidate: str,
    rubric: str = "exact_keyword_overlap",
) -> dict[str, str]:
    """Grade a candidate answer using a deterministic keyword overlap check."""

    if rubric != "exact_keyword_overlap":
        return {"verdict": "FAIL", "reason": "unsupported rubric"}

    reference_words = {
        word.strip(".,;:!?()[]{}").lower()
        for word in reference.split()
        if word.strip(".,;:!?()[]{}")
    }
    candidate_words = {
        word.strip(".,;:!?()[]{}").lower()
        for word in candidate.split()
        if word.strip(".,;:!?()[]{}")
    }
    verdict = "PASS" if reference_words <= candidate_words else "FAIL"
    return {"verdict": verdict}


if __name__ == "__main__":
    mcp.run(transport="stdio")
