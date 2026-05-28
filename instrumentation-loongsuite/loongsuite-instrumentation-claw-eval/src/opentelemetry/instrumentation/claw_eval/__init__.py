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

"""
OpenTelemetry claw-eval Instrumentation
=======================================

Automatic instrumentation for the `claw-eval
<https://github.com/claw-eval/claw-eval>`_ evaluation framework.

Uses **wrapt** monkey-patching to wrap key entry points, the agent loop,
tool dispatchers, compaction, and judge calls that should be suppressed from
producing their own spans — producing a hierarchical trace:

    ENTRY → AGENT → STEP → TOOL / CHAIN

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.claw_eval import ClawEvalInstrumentor

    ClawEvalInstrumentor().instrument()

    # Then run claw-eval as normal (CLI or programmatic)

API
---
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.claw_eval.config import (
    OTEL_INSTRUMENTATION_CLAW_EVAL_ENABLED,
)
from opentelemetry.instrumentation.claw_eval.package import _instruments
from opentelemetry.instrumentation.claw_eval.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

logger = logging.getLogger(__name__)

__all__ = ["ClawEvalInstrumentor"]


def _unwrap_func(module_path: str, func_name: str) -> None:
    """Restore a module-level function wrapped by *wrapt*."""
    try:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, func_name, None)
        if fn is not None and hasattr(fn, "__wrapped__"):
            setattr(mod, func_name, fn.__wrapped__)
    except Exception:
        pass


def _unwrap_method(
    module_path: str, class_name: str, method_name: str
) -> None:
    """Restore a class method wrapped by *wrapt*."""
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls is None:
            return
        meth = getattr(cls, method_name, None)
        if meth is not None and hasattr(meth, "__wrapped__"):
            setattr(cls, method_name, meth.__wrapped__)
    except Exception:
        pass


class ClawEvalInstrumentor(BaseInstrumentor):
    """Instrumentation that adds OpenTelemetry traces to claw-eval.

    Wraps the following symbols via *wrapt*:

    * **ENTRY** — ``cli.cmd_run``, ``cli.cmd_batch``, ``cli._run_single_task``
    * **AGENT** — ``runner.loop.run_task``
    * **STEP** — ``OpenAICompatProvider.chat`` rotates STEP spans
    * **CHAIN** — ``compact.do_auto_compact``
    * **TOOL**  — ``ToolDispatcher.dispatch``, ``SandboxToolDispatcher.dispatch``
    * **Judge (suppress only)** — ``LLMJudge.evaluate``, ``evaluate_actions``,
      ``evaluate_visual``: nested LLM SDK / HTTP spans are suppressed and no
      judge LLM span is emitted, keeping the trace tail clean.
    * **Per-task grader (suppress only)** — ``registry.get_grader`` and
      ``base.load_peer_grader`` are wrapped so any grader class loaded via
      them has its ``_llm_score_classifications`` (and similar evaluation
      helpers) auto-suppressed. This catches the per-task grader code paths
      that talk to ``judge.client.chat.completions.create`` directly,
      bypassing ``LLMJudge.evaluate*``.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not OTEL_INSTRUMENTATION_CLAW_EVAL_ENABLED:
            logger.info("claw-eval instrumentation disabled via env var")
            return

        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__,
            __version__,
            tracer_provider=tracer_provider,
        )

        from opentelemetry.instrumentation.claw_eval.internal.wrappers import (
            DoAutoCompactWrapper,
            EntryWrapper,
            GetGraderWrapper,
            JudgeWrapper,
            LoadPeerGraderWrapper,
            ProviderChatWrapper,
            RunSingleTaskWrapper,
            RunTaskWrapper,
            ToolDispatchWrapper,
        )

        # --- CLI entry points (ENTRY) ---
        for func_name, cmd in [("cmd_run", "run"), ("cmd_batch", "batch")]:
            try:
                wrap_function_wrapper(
                    "claw_eval.cli",
                    func_name,
                    EntryWrapper(tracer, cmd),
                )
            except Exception as exc:
                logger.warning(
                    "Could not wrap claw_eval.cli.%s: %s", func_name, exc
                )

        try:
            wrap_function_wrapper(
                "claw_eval.cli",
                "_run_single_task",
                RunSingleTaskWrapper(tracer),
            )
        except Exception as exc:
            logger.warning("Could not wrap _run_single_task: %s", exc)

        # --- Agent loop (AGENT) ---
        try:
            wrap_function_wrapper(
                "claw_eval.runner.loop",
                "run_task",
                RunTaskWrapper(tracer),
            )
        except Exception as exc:
            logger.warning("Could not wrap run_task: %s", exc)

        # --- Provider chat (STEP rotation) ---
        try:
            wrap_function_wrapper(
                "claw_eval.runner.providers.openai_compat",
                "OpenAICompatProvider.chat",
                ProviderChatWrapper(tracer),
            )
        except Exception as exc:
            logger.warning("Could not wrap OpenAICompatProvider.chat: %s", exc)

        # --- Context compaction (CHAIN) ---
        try:
            wrap_function_wrapper(
                "claw_eval.runner.compact",
                "do_auto_compact",
                DoAutoCompactWrapper(tracer),
            )
        except Exception as exc:
            logger.warning("Could not wrap do_auto_compact: %s", exc)

        # --- Tool dispatchers (TOOL) ---
        try:
            wrap_function_wrapper(
                "claw_eval.runner.dispatcher",
                "ToolDispatcher.dispatch",
                ToolDispatchWrapper(tracer),
            )
        except Exception as exc:
            logger.warning("Could not wrap ToolDispatcher.dispatch: %s", exc)

        try:
            wrap_function_wrapper(
                "claw_eval.runner.sandbox_dispatcher",
                "SandboxToolDispatcher.dispatch",
                ToolDispatchWrapper(tracer),
            )
        except Exception as exc:
            logger.debug(
                "Could not wrap SandboxToolDispatcher.dispatch: %s", exc
            )

        # --- LLM Judge (suppress nested SDK / HTTP spans, no judge span) ---
        for method in ("evaluate", "evaluate_actions", "evaluate_visual"):
            try:
                wrap_function_wrapper(
                    "claw_eval.graders.llm_judge",
                    f"LLMJudge.{method}",
                    JudgeWrapper(tracer, method),
                )
            except Exception as exc:
                logger.warning("Could not wrap LLMJudge.%s: %s", method, exc)

        # --- Per-task grader evaluation helpers ---
        # Per-task ``tasks/T*/grader.py`` defines helpers like
        # ``_llm_score_classifications`` that bypass ``LLMJudge.evaluate*``
        # and call ``judge.client.chat.completions.create`` directly.
        # Hooking the two grader loaders lets us walk each loaded grader's
        # MRO and install span-suppression on those helpers automatically.
        try:
            wrap_function_wrapper(
                "claw_eval.graders.registry",
                "get_grader",
                GetGraderWrapper(tracer),
            )
        except Exception as exc:
            logger.warning("Could not wrap get_grader: %s", exc)

        try:
            wrap_function_wrapper(
                "claw_eval.graders.base",
                "load_peer_grader",
                LoadPeerGraderWrapper(tracer),
            )
        except Exception as exc:
            logger.warning("Could not wrap load_peer_grader: %s", exc)

    def _uninstrument(self, **kwargs: Any) -> None:
        # CLI entry points
        _unwrap_func("claw_eval.cli", "cmd_run")
        _unwrap_func("claw_eval.cli", "cmd_batch")
        _unwrap_func("claw_eval.cli", "_run_single_task")

        # Agent loop
        _unwrap_func("claw_eval.runner.loop", "run_task")

        # Provider chat
        _unwrap_method(
            "claw_eval.runner.providers.openai_compat",
            "OpenAICompatProvider",
            "chat",
        )

        # Context compaction
        _unwrap_func("claw_eval.runner.compact", "do_auto_compact")

        # Tool dispatchers
        _unwrap_method(
            "claw_eval.runner.dispatcher",
            "ToolDispatcher",
            "dispatch",
        )
        _unwrap_method(
            "claw_eval.runner.sandbox_dispatcher",
            "SandboxToolDispatcher",
            "dispatch",
        )

        # LLM Judge
        for method in ("evaluate", "evaluate_actions", "evaluate_visual"):
            _unwrap_method(
                "claw_eval.graders.llm_judge",
                "LLMJudge",
                method,
            )

        # Per-task grader loaders. Note: dynamically wrapped per-task
        # ``_llm_score_classifications`` methods on already-loaded grader
        # classes are intentionally not unwrapped here — those modules are
        # loaded under synthetic names like ``task_grader_<id>`` and there
        # is no stable handle to walk. Unwrapping the loaders is enough to
        # stop *new* graders from getting wrapped after uninstrument.
        _unwrap_func("claw_eval.graders.registry", "get_grader")
        _unwrap_func("claw_eval.graders.base", "load_peer_grader")
