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

"""Map BFCL ``ModelStyle`` enum values to ``gen_ai.provider.name``."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from opentelemetry.instrumentation.bfclv4.internal.attributes import (
    BFCL_OSS_BACKEND,
)

# The BFCL backend name (vllm / sglang / ...) is communicated from the ENTRY
# wrapper to the per-thread STEP/AGENT wrappers via this env var.  The ENTRY
# wrapper writes to it before invoking the wrapped function and clears it in
# the ``finally`` clause.
OSS_BACKEND_ENV = "BFCL_BACKEND"


def infer_provider(handler: Any) -> Tuple[str, Dict[str, Any]]:
    """Return ``(provider_name, extra_attributes)`` for a BFCL handler.

    Falls back to ``"unknown"`` if BFCL is not importable or if the handler
    has no ``model_style`` attribute.
    """

    try:
        from bfcl_eval.constants.enums import (  # noqa: PLC0415
            ModelStyle,
        )
    except ImportError:
        return "unknown", {}

    style = getattr(handler, "model_style", None)
    if style is None:
        return "unknown", {}

    if style is ModelStyle.OSSMODEL:
        backend = (os.getenv(OSS_BACKEND_ENV) or "").lower()
        if backend in ("vllm", "sglang"):
            return backend, {BFCL_OSS_BACKEND: backend}
        return "oss", {BFCL_OSS_BACKEND: "unknown"}

    mapping = {
        ModelStyle.OPENAI_COMPLETIONS: "openai",
        ModelStyle.OPENAI_RESPONSES: "openai",
        ModelStyle.ANTHROPIC: "anthropic",
        ModelStyle.GOOGLE: "gcp.gemini",
        ModelStyle.MISTRAL: "mistral_ai",
        ModelStyle.COHERE: "cohere",
        ModelStyle.AMAZON: "aws.bedrock",
        ModelStyle.FIREWORK_AI: "fireworks_ai",
        ModelStyle.WRITER: "writer",
        ModelStyle.NOVITA_AI: "novita",
        ModelStyle.NEXUS: "nexusflow",
        ModelStyle.GORILLA: "gorilla",
    }
    return mapping.get(style, "unknown"), {}
