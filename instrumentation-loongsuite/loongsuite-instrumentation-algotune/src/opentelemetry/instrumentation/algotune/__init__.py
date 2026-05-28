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
OpenTelemetry AlgoTune Instrumentation
======================================

Automatic instrumentation for the `AlgoTune
<https://github.com/oripress/AlgoTune>`_ benchmark framework.

This instrumentor produces the AlgoTune-business span tree
(``ENTRY`` / ``AGENT`` / ``STEP`` / ``TOOL`` / ``TASK``) and intentionally
**does not** create LLM spans for the LiteLLM call path. Those are
expected to be produced by an already-loaded LiteLLM instrumentor (e.g.
``opentelemetry-instrumentation-litellm`` or
``openinference-instrumentation-litellm``); they automatically become
children of the active ``STEP`` span thanks to OpenTelemetry context
propagation.

A separate, **opt-in** wrapper exists for ``TogetherModel.query``, which
hits the Together API directly via ``requests.post`` and is therefore
not covered by the LiteLLM instrumentor. Enable it with the environment
variable ``ALGOTUNE_OTEL_INSTRUMENT_TOGETHER=true``.

Span hierarchy
--------------

::

    ENTRY: enter_ai_application_system          ← AlgoTuner.main:main()
    └── AGENT: invoke_agent AlgoTuner           ← LLMInterface.run_task()
        ├── STEP: react step  [round=N]         ← get_response + handle_function_call
        │   ├── LLM:  chat <model>              ← LiteLLM instrumentor (auto)
        │   │                                     OR TogetherModel.query (this pkg)
        │   └── TOOL: execute_tool <command>    ← CommandHandlers.handle_command
        │       └── TASK: run_task benchmark.dataset_eval ← _runner_eval_dataset
        │           ├── TASK: run_task benchmark.baseline_generation ← get_baseline_times
        │           └── TASK: run_task benchmark.problem_eval [×N] ← evaluate_single
        └── ...

Usage
-----

.. code:: python

    # 1) Load the LiteLLM instrumentor first so LLM spans are produced.
    from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
    LiteLLMInstrumentor().instrument()

    # 2) Then load the AlgoTune instrumentor for business spans.
    from opentelemetry.instrumentation.algotune import AlgoTuneInstrumentor
    AlgoTuneInstrumentor().instrument()

    # Run AlgoTune as normal.
    # python -m AlgoTuner.main --model gpt-4o --task tsp

Configuration
-------------

Environment variables:

* ``OTEL_INSTRUMENTATION_ALGOTUNE_ENABLED`` — master enable switch (default ``true``).
* ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` — capture
  tool-call arguments / result messages (default ``false``).
* ``ALGOTUNE_OTEL_MAX_CONTENT_LENGTH`` — character truncation for string
  attributes (default ``4096``).
* ``ALGOTUNE_OTEL_INSTRUMENT_TOGETHER`` — wrap ``TogetherModel.query`` with
  a manual LLM span (default ``false``).

API
---
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.algotune.config import (
    ALGOTUNE_OTEL_INSTRUMENT_TOGETHER,
    OTEL_INSTRUMENTATION_ALGOTUNE_ENABLED,
)
from opentelemetry.instrumentation.algotune.package import _instruments
from opentelemetry.instrumentation.algotune.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

logger = logging.getLogger(__name__)

__all__ = ["AlgoTuneInstrumentor"]


# Patch sites are (module_path, attribute_name) tuples. We use the source
# module so that the wrap survives import-order changes.
_PATCH_SITES: list[tuple[str, str, str]] = [
    # (logical_name, module_path, qualified_attribute)
    ("main", "AlgoTuner.main", "main"),
    (
        "run_task",
        "AlgoTuner.interfaces.llm_interface",
        "LLMInterface.run_task",
    ),
    (
        "get_response",
        "AlgoTuner.interfaces.llm_interface",
        "LLMInterface.get_response",
    ),
    (
        "handle_function_call",
        "AlgoTuner.interfaces.llm_interface",
        "LLMInterface.handle_function_call",
    ),
    (
        "handle_command",
        "AlgoTuner.interfaces.commands.handlers",
        "CommandHandlers.handle_command",
    ),
    (
        "_runner_eval_dataset",
        "AlgoTuner.interfaces.commands.handlers",
        "CommandHandlers._runner_eval_dataset",
    ),
    (
        "evaluate_single",
        "AlgoTuner.utils.evaluator.evaluation_orchestrator",
        "EvaluationOrchestrator.evaluate_single",
    ),
    (
        "get_baseline_times",
        "AlgoTuner.utils.evaluator.baseline_manager",
        "BaselineManager.get_baseline_times",
    ),
    ("query", "AlgoTuner.models.lite_llm_model", "LiteLLMModel.query"),
    (
        "_execute_query",
        "AlgoTuner.models.lite_llm_model",
        "LiteLLMModel._execute_query",
    ),
]

_TOGETHER_PATCH_SITE: tuple[str, str, str] = (
    "together_query",
    "AlgoTuner.models.together_model",
    "TogetherModel.query",
)


def _safe_wrap(module_path: str, name: str, wrapper: Any) -> bool:
    """Wrap ``module_path.name`` with ``wrapper``; swallow ImportError."""
    try:
        wrap_function_wrapper(module_path, name, wrapper)
        return True
    except (ImportError, AttributeError) as exc:
        logger.debug(
            "AlgoTune: skipping wrap %s.%s (%s)", module_path, name, exc
        )
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "AlgoTune: could not wrap %s.%s: %s", module_path, name, exc
        )
        return False


def _safe_unwrap(module_path: str, qualname: str) -> None:
    """Restore an attribute wrapped by ``wrapt``.

    ``qualname`` may be ``"Class.method"`` or just ``"func"``. We walk the
    module/class chain and restore via ``__wrapped__`` when present.
    """
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return

    parts = qualname.split(".")
    parent: Any = mod
    for part in parts[:-1]:
        parent = getattr(parent, part, None)
        if parent is None:
            return
    leaf_name = parts[-1]
    leaf = getattr(parent, leaf_name, None)
    if leaf is None:
        return
    original = getattr(leaf, "__wrapped__", None)
    if original is None:
        return
    try:
        setattr(parent, leaf_name, original)
    except Exception:  # noqa: BLE001
        pass


class AlgoTuneInstrumentor(BaseInstrumentor):
    """An instrumentor for the AlgoTune benchmark framework.

    Covers six AlgoTune-business span kinds:

    * **ENTRY** – ``AlgoTuner.main.main``
    * **AGENT** – ``LLMInterface.run_task``
    * **STEP**  – ``LLMInterface.get_response`` (open) +
      ``LLMInterface.handle_function_call`` (close)
    * **TOOL**  – ``CommandHandlers.handle_command``
    * **TASK**  – ``CommandHandlers._runner_eval_dataset``,
      ``EvaluationOrchestrator.evaluate_single``,
      ``BaselineManager.get_baseline_times``

    The LiteLLM call path (``LiteLLMModel.query`` / ``_execute_query``)
    is wrapped only to publish ``algo.llm.retry_count`` onto the active
    STEP span; **no LLM span is created**. LLM spans for that path are
    expected from a separately-loaded LiteLLM instrumentor.

    The ``TogetherModel.query`` bypass (raw HTTP, not via ``litellm``) is
    only wrapped when ``ALGOTUNE_OTEL_INSTRUMENT_TOGETHER=true``.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not OTEL_INSTRUMENTATION_ALGOTUNE_ENABLED:
            logger.info("AlgoTune instrumentation disabled via env var")
            return

        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__,
            __version__,
            tracer_provider=tracer_provider,
        )

        from opentelemetry.instrumentation.algotune.internal.wrappers import (
            EvaluateSingleWrapper,
            GetBaselineTimesWrapper,
            GetResponseWrapper,
            HandleCommandWrapper,
            HandleFunctionCallWrapper,
            LiteLLMExecuteQueryWrapper,
            LiteLLMQueryWrapper,
            MainWrapper,
            RunnerEvalDatasetWrapper,
            RunTaskWrapper,
            TogetherModelQueryWrapper,
        )

        wrappers_by_name: dict[str, Any] = {
            "main": MainWrapper(tracer),
            "run_task": RunTaskWrapper(tracer),
            "get_response": GetResponseWrapper(tracer),
            "handle_function_call": HandleFunctionCallWrapper(),
            "handle_command": HandleCommandWrapper(tracer),
            "_runner_eval_dataset": RunnerEvalDatasetWrapper(tracer),
            "evaluate_single": EvaluateSingleWrapper(tracer),
            "get_baseline_times": GetBaselineTimesWrapper(tracer),
            "query": LiteLLMQueryWrapper(),
            "_execute_query": LiteLLMExecuteQueryWrapper(),
        }

        for logical_name, module_path, qualname in _PATCH_SITES:
            wrapper = wrappers_by_name.get(logical_name)
            if wrapper is None:
                continue
            if not _safe_wrap(module_path, qualname, wrapper):
                logger.info(
                    "AlgoTune: %s not yet importable; skipping wrap",
                    f"{module_path}.{qualname}",
                )

        if ALGOTUNE_OTEL_INSTRUMENT_TOGETHER:
            logical, module_path, qualname = _TOGETHER_PATCH_SITE
            _safe_wrap(
                module_path,
                qualname,
                TogetherModelQueryWrapper(tracer),
            )

        # Best-effort sanity check: warn if no LiteLLM instrumentor is
        # loaded -- the trace tree will still be valid but LLM spans will
        # be missing.
        if not _is_litellm_instrumented():
            logger.warning(
                "AlgoTune instrumentation: litellm.completion does not look"
                " instrumented. LLM spans will be missing from the trace"
                " tree. Load opentelemetry-instrumentation-litellm (or"
                " openinference-instrumentation-litellm) before AlgoTune"
                " starts."
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        for _logical, module_path, qualname in _PATCH_SITES:
            _safe_unwrap(module_path, qualname)
        _logical, module_path, qualname = _TOGETHER_PATCH_SITE
        _safe_unwrap(module_path, qualname)


def _is_litellm_instrumented() -> bool:
    """Return ``True`` iff ``litellm.completion`` appears to be wrapped.

    We look for the ``__wrapped__`` attribute set by ``wrapt`` /
    ``functools.wraps``. Returns ``False`` (no warning suppressed) when
    ``litellm`` itself is not importable -- in that case AlgoTune will
    fail before we get a chance to emit spans anyway.
    """
    try:
        import litellm  # noqa: PLC0415
    except ImportError:
        return False
    completion = getattr(litellm, "completion", None)
    if completion is None:
        return False
    return hasattr(completion, "__wrapped__")
