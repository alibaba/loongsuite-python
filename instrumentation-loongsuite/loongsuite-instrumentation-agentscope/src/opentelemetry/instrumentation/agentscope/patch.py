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

"""Patch functions for AgentScope instrumentation."""

from __future__ import annotations

import json
import logging
import os
import timeit
from importlib import import_module
from pathlib import Path

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_span_utils import (
    _apply_execute_tool_finish_attributes,
)
from opentelemetry.util.genai.extended_types import ExecuteToolInvocation
from opentelemetry.util.genai.span_utils import _apply_error_attributes
from opentelemetry.util.genai.types import Error

from .utils import (
    apply_entry_baggage_identity,
    entry_baggage_identity_attributes,
)

logger = logging.getLogger(__name__)


# The canonical filename that defines a skill.  Only reads targeting this
# file are treated as "skill load" operations.
_SKILL_MANIFEST = "SKILL.md"


def _resolves_to_skill_md(
    file_path: str,
    skill_dir: str,
    *,
    allow_bare: bool = False,
) -> bool:
    """Return True if *file_path* points to the ``SKILL.md`` inside *skill_dir*.

    Supports three path forms that appear in real CoPaw/QwenPaw traces:

    1. **Absolute path** – e.g.
       ``/home/user/.copaw/workspaces/default/skills/pdf/SKILL.md``
       Compared directly against ``{skill_dir}/SKILL.md`` using
       ``realpath`` to resolve symlinks (important on macOS where
       ``/var`` → ``/private/var``).

    2. **Bare filename** – ``SKILL.md``
       Only accepted when *allow_bare* is ``True``.  The caller
       (``_match_skill_for_tool``) sets this flag only when exactly
       one skill is registered, so a bare ``SKILL.md`` is unambiguous.
       With multiple skills a bare name could match any of them, so we
       conservatively reject it and let the upstream CoPaw context
       provide the authoritative match instead.

    3. **Workspace-relative path** – e.g. ``skills/pdf/SKILL.md`` or
       ``workspaces/default/skills/pdf/SKILL.md``
       CoPaw resolves these against a workspace / WORKING_DIR that may
       differ from the process cwd.  Instead of guessing the base
       directory, we use **suffix matching**: the normalised path must
       end with ``/{skill_dir_basename}/SKILL.md`` where
       *skill_dir_basename* is the last component of *skill_dir*.
       This avoids depending on cwd at all.
    """
    try:
        # Use realpath to resolve symlinks (e.g. /var → /private/var on macOS)
        real_skill_dir = os.path.normpath(os.path.realpath(skill_dir))
        expected = os.path.join(real_skill_dir, _SKILL_MANIFEST)

        # --- Try 1: absolute path ---
        if os.path.isabs(file_path):
            candidate = os.path.normpath(os.path.realpath(file_path))
            return candidate == expected

        normalised_rel = os.path.normpath(file_path)

        # --- Try 2: bare "SKILL.md" (only when unambiguous) ---
        if normalised_rel == _SKILL_MANIFEST:
            return allow_bare

        # --- Try 3: workspace-relative path (suffix matching) ---
        # Build the expected suffix: e.g. "pdf/SKILL.md"
        skill_dir_basename = os.path.basename(real_skill_dir)
        expected_suffix = os.path.join(skill_dir_basename, _SKILL_MANIFEST)

        # The relative path must end with exactly "{skill_name}/SKILL.md"
        # Use os.sep-aware comparison to avoid partial-name collisions
        # (e.g. "pdf-extra/SKILL.md" must NOT match skill dir "pdf")
        if normalised_rel == expected_suffix:
            return True
        if normalised_rel.endswith(os.sep + expected_suffix):
            return True

    except Exception:
        return False

    return False


def _enrich_skill_metadata(skill):
    """Enrich a matched skill dict with version and id from SKILL.md.

    AgentScope's ``AgentSkill`` only stores ``name``, ``description``,
    and ``dir``. This function best-effort extracts ``version`` from
    SKILL.md/frontmatter-related sources and builds a
    **runtime / deployment-scoped** ``id`` when ``dir`` is available.

    The enrichment is best-effort: if frontmatter/version data cannot be
    read (e.g. file missing, parse error, or running without CoPaw), the
    returned dict may still include a path-derived ``id`` when possible,
    but ``version`` is only added when successfully determined.

    When CoPaw is available, its ``_read_frontmatter_safe`` and
    ``_extract_version`` helpers are preferred because they are the
    canonical source of truth.  Otherwise we fall back to a lightweight
    ``frontmatter`` parse.

    Returns a new dict (never mutates the original).
    """
    skill_dir = skill.get("dir", "")
    skill_name = skill.get("name", "")
    if not skill_dir:
        return dict(skill)

    enriched = dict(skill)

    # --- Extract version ---
    version_text = None

    # Prefer CoPaw's own helpers (canonical source of truth). If that
    # path fails at runtime for any reason, continue to lightweight
    # frontmatter/manifest fallbacks instead of dropping version entirely.
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

            skill_md_path = os.path.join(skill_dir, _SKILL_MANIFEST)
            with open(skill_md_path, "r", encoding="utf-8") as fh:
                post = frontmatter.load(fh)
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
                # CoPaw-specific runtime fallback: workspace skill.json may
                # already contain normalized version_text even when direct
                # helper/frontmatter extraction is unavailable in-process.
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

    # --- Build runtime/deployment-scoped skill ID ---
    # Format: workspace:{workspace_name}:{skill_name}
    # NOT a globally canonical ID — stable within one workspace.
    try:
        parts = skill_dir.replace("\\", "/").split("/")
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


def _match_skill_for_tool(instance, tool_args):
    """Detect if a tool execution is loading a skill.

    Returns an enriched dict with skill metadata (including version and
    id when available) when **all** of the following conditions are met:

    1. The Toolkit instance has registered skills.
    2. The tool arguments contain a file path (``file_path``, ``path``,
       or ``target_file``).
    3. That path resolves to the ``SKILL.md`` file at the top level of a
       registered skill directory.

    Only reads of ``SKILL.md`` are considered skill loads.  Accessing
    other files inside a skill directory (scripts, references, resources)
    does **not** trigger skill attributes, keeping the semantic boundary
    clean.

    The skill judgment is precise: only skills registered in
    ``toolkit.skills`` (i.e. effective skills for the current workspace
    and channel) can match.  Arbitrary SKILL.md files outside the
    registered set are never matched.

    Otherwise ``None`` is returned.
    """
    if not hasattr(instance, "skills") or not instance.skills:
        return None

    # Extract candidate file path from tool arguments
    file_path = None
    if isinstance(tool_args, dict):
        file_path = (
            tool_args.get("file_path")
            or tool_args.get("path")
            or tool_args.get("target_file")
        )
    if not file_path:
        return None

    # Bare "SKILL.md" is ambiguous when multiple skills are registered.
    # Only allow it when exactly one skill exists.
    single_skill = len(instance.skills) == 1

    matches = []
    for skill in instance.skills.values():
        skill_dir = skill.get("dir", "")
        if not skill_dir:
            continue

        if _resolves_to_skill_md(
            file_path, skill_dir, allow_bare=single_skill
        ):
            matches.append(_enrich_skill_metadata(skill))

    if len(matches) == 1:
        return matches[0]

    return None


def _get_tool_description(instance, tool_name):
    """Get tool description from toolkit."""
    if (
        not tool_name
        or not hasattr(instance, "tools")
        or not isinstance(instance.tools, dict)
    ):
        return None

    tool_obj = instance.tools.get(tool_name)
    if not tool_obj:
        return None

    # First try to get from json_schema (the correct way for AgentScope tools)
    json_schema = getattr(tool_obj, "json_schema", None)
    if isinstance(json_schema, dict):
        func_dict = json_schema.get("function", {})
        if isinstance(func_dict, dict):
            description = func_dict.get("description")
            if description:
                return description

    # Fallback to direct description attribute
    return getattr(tool_obj, "description", None)


def _get_tool_result(chunk):
    """Extract tool result from chunk."""
    if chunk is None:
        return None
    if hasattr(chunk, "content"):
        return chunk.content
    return chunk


async def _trace_async_generator_wrapper(
    result_generator, invocation, span, handler
):
    """
    Async generator wrapper that traces tool execution.

    This function wraps the async generator returned by call_tool_function,
    collects the last chunk, and applies handler's logic without context management.

    Args:
        result_generator: The async generator to wrap (yields ToolResponse objects)
        invocation: ExecuteToolInvocation object to track tool execution data
        span: The OpenTelemetry span (managed by us, not handler)
        handler: ExtendedTelemetryHandler for accessing utility functions
    """
    has_error = False
    last_chunk = None
    error_obj = None

    try:
        async for chunk in result_generator:
            last_chunk = chunk
            yield chunk
    except Exception as e:
        has_error = True
        error_obj = Error(message=str(e), type=type(e))
        raise e from None

    finally:
        # Update invocation with result data
        if not has_error and last_chunk:
            try:
                result_content = _get_tool_result(last_chunk)
                if result_content:
                    invocation.tool_call_result = result_content
            except Exception:
                pass

        # Apply handler's attribute logic (without context management)
        try:
            _apply_execute_tool_finish_attributes(span, invocation)

            if has_error and error_obj:
                _apply_error_attributes(span, error_obj)
                # Record metrics with error
                if handler._metrics_recorder is not None:
                    handler._metrics_recorder.record_extended(
                        span,
                        invocation,
                        error_type=error_obj.type.__qualname__,
                    )
            else:
                # Record metrics without error
                if handler._metrics_recorder is not None:
                    handler._metrics_recorder.record_extended(span, invocation)
        except Exception:
            # Don't let finalization errors break the generator
            pass

        # End the span (we manage it, not handler)
        span.end()


async def wrap_tool_call(wrapped, instance, args, kwargs, handler):
    """
    Async wrapper for Toolkit.call_tool_function.

    Args:
        wrapped: The original async generator function being wrapped
        instance: The Toolkit instance
        args: Positional arguments (tool_call dict, ...)
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (required)
    """

    # Extract tool call information
    tool_call = args[0] if args else kwargs.get("tool_call", {})
    tool_name = (
        tool_call.get("name", "unknown_tool")
        if isinstance(tool_call, dict)
        else "unknown_tool"
    )
    tool_id = tool_call.get("id") if isinstance(tool_call, dict) else None
    tool_args = (
        tool_call.get("input", {}) if isinstance(tool_call, dict) else {}
    )

    # Get tool description from AgentScope's toolkit
    tool_description = _get_tool_description(instance, tool_name)

    # Detect if this tool execution is loading a skill
    matched_skill = _match_skill_for_tool(instance, tool_args)

    # Create invocation object with all tool data
    invocation = ExecuteToolInvocation(
        tool_name=tool_name,
        tool_call_id=tool_id,
        tool_description=tool_description,
        tool_call_arguments=tool_args,
    )
    apply_entry_baggage_identity(invocation)

    # --- Skill attributes ---
    #
    # ``_match_skill_for_tool`` checks whether this tool call reads a
    # registered skill's SKILL.md.  Only skills in ``toolkit.skills``
    # (i.e. effective skills for the current workspace/channel) can match.
    #
    # When a match is found, ``_enrich_skill_metadata`` reads the
    # SKILL.md frontmatter to extract version and builds a runtime-scoped
    # skill_id.  If CoPaw is available, its canonical helpers are used;
    # otherwise a lightweight frontmatter parse provides a best-effort
    # fallback.  When running AgentScope standalone (no CoPaw, no
    # frontmatter lib), name and description are still available from
    # the AgentSkill dict.

    if matched_skill is not None:
        invocation.skill_name = matched_skill.get("name")
        invocation.skill_id = matched_skill.get("id")
        invocation.skill_description = matched_skill.get("description")
        invocation.skill_version = matched_skill.get("version")

    invocation.monotonic_start_s = timeit.default_timer()

    span_name = f"{GenAIAttributes.GenAiOperationNameValues.EXECUTE_TOOL.value} {tool_name}"
    with handler._tracer.start_as_current_span(
        name=span_name,
        kind=SpanKind.INTERNAL,
        end_on_exit=False,
    ) as span:
        try:
            result_generator = await wrapped(*args, **kwargs)
            # Wrap the async generator to collect results and end span when done
            return _trace_async_generator_wrapper(
                result_generator, invocation, span, handler
            )
        except Exception as error:
            # Handle errors before returning the generator
            error_obj = Error(message=str(error), type=type(error))
            _apply_execute_tool_finish_attributes(span, invocation)
            _apply_error_attributes(span, error_obj)

            # Record metrics with error
            if handler._metrics_recorder is not None:
                handler._metrics_recorder.record_extended(
                    span, invocation, error_type=error_obj.type.__qualname__
                )

            span.end()
            raise error from None


async def wrap_formatter_format(wrapped, instance, args, kwargs, tracer=None):
    """
    Async wrapper for TruncatedFormatterBase.format.

    This is a simple operation so we keep the direct tracer approach.

    Args:
        wrapped: The original async function being wrapped
        instance: The TruncatedFormatterBase instance
        args: Positional arguments (msgs)
        kwargs: Keyword arguments
        tracer: OpenTelemetry tracer
    """
    if tracer is None:
        return await wrapped(*args, **kwargs)

    # Use simplified span creation (formatter is an auxiliary operation, doesn't need full GenAI attributes)
    with tracer.start_as_current_span("format_messages") as span:
        try:
            # Record only basic information
            span.set_attribute("gen_ai.operation.name", "format")
            span.set_attributes(entry_baggage_identity_attributes())

            # Execute the wrapped async call
            result = await wrapped(*args, **kwargs)

            return result

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
