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

"""Focused tests for AgentScope skill detection and span attributes."""

from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.agentscope.patch import (
    _enrich_skill_metadata,
    _match_skill_for_tool,
    _resolves_to_skill_md,
    _trace_async_generator_wrapper,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SKILL_DESCRIPTION,
    GEN_AI_SKILL_ID,
    GEN_AI_SKILL_NAME,
    GEN_AI_SKILL_VERSION,
)
from opentelemetry.util.genai.extended_types import ExecuteToolInvocation


def _make_instance(skills: dict | None = None):
    instance = SimpleNamespace()
    if skills is not None:
        instance.skills = skills
    return instance


class _FakeSpan:
    def __init__(self):
        self.attributes = {}
        self.ended = False

    def set_attributes(self, attrs: dict):
        self.attributes.update(attrs)

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def update_name(self, name: str):
        del name

    def end(self):
        self.ended = True

    def set_status(self, status):
        del status

    def record_exception(self, exception):
        del exception


class _FakeHandler:
    def __init__(self):
        self._metrics_recorder = None


class TestResolvesToSkillMd:
    def test_absolute_skill_md_matches(self, tmp_path):
        skill_dir = tmp_path / "skills" / "pdf"
        skill_dir.mkdir(parents=True)
        target = skill_dir / "SKILL.md"
        target.touch()

        assert _resolves_to_skill_md(str(target), str(skill_dir)) is True

    def test_non_skill_file_rejected(self, tmp_path):
        skill_dir = tmp_path / "skills" / "pdf"
        skill_dir.mkdir(parents=True)
        target = skill_dir / "scripts" / "run.sh"
        target.parent.mkdir(parents=True)
        target.touch()

        assert _resolves_to_skill_md(str(target), str(skill_dir)) is False

    def test_prefix_collision_rejected(self, tmp_path):
        skill_dir = tmp_path / "skills" / "pdf"
        skill_dir.mkdir(parents=True)
        target_dir = tmp_path / "skills" / "pdf-extra"
        target_dir.mkdir(parents=True)
        target = target_dir / "SKILL.md"
        target.touch()

        assert _resolves_to_skill_md(str(target), str(skill_dir)) is False

    def test_workspace_relative_match(self, tmp_path):
        skill_dir = tmp_path / "workspaces" / "default" / "skills" / "pdf"
        skill_dir.mkdir(parents=True)

        assert (
            _resolves_to_skill_md("skills/pdf/SKILL.md", str(skill_dir))
            is True
        )

    def test_bare_skill_md_requires_allow_bare(self, tmp_path):
        skill_dir = tmp_path / "skills" / "pdf"
        skill_dir.mkdir(parents=True)

        assert _resolves_to_skill_md("SKILL.md", str(skill_dir)) is False
        assert (
            _resolves_to_skill_md("SKILL.md", str(skill_dir), allow_bare=True)
            is True
        )


class TestMatchSkillForTool:
    def test_matches_registered_skill_manifest(self, tmp_path):
        skill_dir = tmp_path / "workspaces" / "default" / "skills" / "news"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "---\n"
            "name: news\n"
            "description: Latest news\n"
            "metadata:\n"
            "  builtin_skill_version: 1.0\n"
            "---\n"
        )

        instance = _make_instance(
            {
                "news": {
                    "name": "news",
                    "description": "Latest news",
                    "dir": str(skill_dir),
                }
            }
        )

        result = _match_skill_for_tool(
            instance,
            {"file_path": str(skill_md)},
        )

        assert result is not None
        assert result["name"] == "news"
        assert result["id"] == "workspace:default:news"
        assert result["version"] == "1.0"

    def test_non_manifest_read_does_not_match(self, tmp_path):
        skill_dir = tmp_path / "skills" / "news"
        skill_dir.mkdir(parents=True)
        target = skill_dir / "references" / "note.md"
        target.parent.mkdir(parents=True)
        target.touch()

        instance = _make_instance(
            {"news": {"name": "news", "dir": str(skill_dir)}}
        )

        assert (
            _match_skill_for_tool(instance, {"file_path": str(target)}) is None
        )

    def test_bare_skill_md_only_matches_single_skill(self, tmp_path):
        one_dir = tmp_path / "skills" / "one"
        one_dir.mkdir(parents=True)
        (one_dir / "SKILL.md").touch()

        single = _make_instance({"one": {"name": "one", "dir": str(one_dir)}})
        multiple = _make_instance(
            {
                "one": {"name": "one", "dir": str(one_dir)},
                "two": {
                    "name": "two",
                    "dir": str(tmp_path / "skills" / "two"),
                },
            }
        )

        assert (
            _match_skill_for_tool(single, {"file_path": "SKILL.md"})
            is not None
        )
        assert (
            _match_skill_for_tool(multiple, {"file_path": "SKILL.md"}) is None
        )

    def test_workspace_relative_match_returns_none_when_ambiguous(
        self, tmp_path
    ):
        default_dir = tmp_path / "workspaces" / "default" / "skills" / "pdf"
        demo_dir = tmp_path / "workspaces" / "demo" / "skills" / "pdf"
        default_dir.mkdir(parents=True)
        demo_dir.mkdir(parents=True)
        (default_dir / "SKILL.md").touch()
        (demo_dir / "SKILL.md").touch()

        instance = _make_instance(
            {
                "default_pdf": {"name": "pdf", "dir": str(default_dir)},
                "demo_pdf": {"name": "pdf", "dir": str(demo_dir)},
            }
        )

        assert (
            _match_skill_for_tool(
                instance, {"file_path": "skills/pdf/SKILL.md"}
            )
            is None
        )


class TestEnrichSkillMetadata:
    def test_reads_version_from_frontmatter(self, tmp_path):
        skill_dir = tmp_path / "workspaces" / "demo" / "skills" / "writer"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: writer\ndescription: Write docs\nversion: 2.1.0\n---\n"
        )

        skill = {
            "name": "writer",
            "description": "Write docs",
            "dir": str(skill_dir),
        }
        enriched = _enrich_skill_metadata(skill)

        assert enriched["version"] == "2.1.0"
        assert enriched["id"] == "workspace:demo:writer"
        assert "version" not in skill
        assert "id" not in skill

    def test_falls_back_to_workspace_skill_json_version(self, tmp_path):
        workspace_dir = tmp_path / "workspaces" / "demo"
        skill_dir = workspace_dir / "skills" / "writer"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: writer\ndescription: Write docs\n---\n"
        )
        (workspace_dir / "skill.json").write_text(
            ('{"skills":{"writer":{"metadata":{"version_text":"3.0.0"}}}}')
        )

        enriched = _enrich_skill_metadata(
            {
                "name": "writer",
                "description": "Write docs",
                "dir": str(skill_dir),
            }
        )

        assert enriched["version"] == "3.0.0"
        assert enriched["id"] == "workspace:demo:writer"

    def test_missing_skill_md_is_best_effort(self, tmp_path):
        skill_dir = tmp_path / "workspaces" / "default" / "skills" / "writer"
        skill_dir.mkdir(parents=True)

        enriched = _enrich_skill_metadata(
            {
                "name": "writer",
                "description": "Write docs",
                "dir": str(skill_dir),
            }
        )

        assert enriched["id"] == "workspace:default:writer"
        assert "version" not in enriched


class TestSkillSpanAttributes:
    @pytest.mark.asyncio
    async def test_trace_async_generator_wrapper_preserves_skill_attributes(
        self,
    ):
        async def fake_tool_generator():
            yield "chunk"

        invocation = ExecuteToolInvocation(
            tool_name="read_file",
            tool_type="function",
            skill_name="news",
            skill_id="workspace:default:news",
            skill_description="Latest news",
            skill_version="1.0",
        )
        span = _FakeSpan()
        handler = _FakeHandler()

        chunks = []
        async for chunk in _trace_async_generator_wrapper(
            fake_tool_generator(), invocation, span, handler
        ):
            chunks.append(chunk)

        assert chunks == ["chunk"]
        assert span.attributes[GEN_AI_SKILL_NAME] == "news"
        assert span.attributes[GEN_AI_SKILL_ID] == "workspace:default:news"
        assert span.attributes[GEN_AI_SKILL_DESCRIPTION] == "Latest news"
        assert span.attributes[GEN_AI_SKILL_VERSION] == "1.0"
        assert span.ended is True
