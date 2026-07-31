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

"""AgentScope-version-independent skill metadata helpers."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

_SKILL_MANIFEST = "SKILL.md"


def _enrich_skill_metadata(skill: dict[str, Any]) -> dict[str, Any]:
    """Enrich a skill mapping with a version and runtime-scoped identifier."""

    skill_dir = skill.get("dir", "")
    skill_name = skill.get("name", "")
    if not skill_dir:
        return dict(skill)

    enriched = dict(skill)
    version_text = None

    try:
        skills_manager = import_module("copaw.agents.skills_manager")
        post = skills_manager._read_frontmatter_safe(
            Path(skill_dir), skill_name
        )
        version_text = skills_manager._extract_version(post)
    except Exception:
        version_text = None

    if not version_text:
        try:
            frontmatter = import_module("frontmatter")
            post = frontmatter.load(Path(skill_dir) / _SKILL_MANIFEST)
            metadata = post.get("metadata") or {}
            for value in (
                post.get("version"),
                metadata.get("version"),
                metadata.get("builtin_skill_version"),
            ):
                if value not in (None, ""):
                    version_text = str(value)
                    break
        except Exception:
            version_text = None

    if not version_text:
        try:
            skill_path = Path(skill_dir)
            skill_json_path = skill_path.parent.parent / "skill.json"
            if skill_json_path.exists():
                payload = json.loads(
                    skill_json_path.read_text(encoding="utf-8")
                )
                entry = payload.get("skills", {}).get(skill_name, {})
                metadata = entry.get("metadata", {}) or {}
                value = metadata.get("version_text")
                if value not in (None, ""):
                    version_text = str(value)
        except Exception:
            version_text = None

    if version_text:
        enriched["version"] = str(version_text)

    try:
        parts = str(skill_dir).replace("\\", "/").split("/")
        workspace_name = "default"
        try:
            skills_idx = len(parts) - 1 - parts[::-1].index("skills")
            if skills_idx >= 1:
                workspace_name = parts[skills_idx - 1]
        except ValueError:
            pass
        enriched["id"] = f"workspace:{workspace_name}:{skill_name}"
    except Exception:
        pass

    return enriched
