"""Constant attribute keys & framework identity used across wrappers."""

from __future__ import annotations

GEN_AI_FRAMEWORK = "gen_ai.framework"
GEN_AI_SPAN_KIND = "gen_ai.span.kind"

FRAMEWORK_NAME = "openhands"

# OpenHands-specific span attributes (namespaced to avoid clashing with the
# generic GenAI semconv attributes already provided by upstream).
OH_INITIAL_MESSAGE_PREVIEW = "openhands.initial_message.preview"
