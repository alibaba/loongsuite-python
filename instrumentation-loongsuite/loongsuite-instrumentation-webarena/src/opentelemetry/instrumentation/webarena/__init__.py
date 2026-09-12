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
OpenTelemetry WebArena Instrumentation
======================================

Automatic instrumentation for the
`WebArena <https://github.com/web-arena-x/webarena>`_ benchmark framework.

Span hierarchy
--------------

::

    ENTRY  webarena_task                       (per task; ScriptBrowserEnv.reset)
    └── CHAIN  workflow webarena_task          (same lifecycle as ENTRY)
         ├── STEP  react step                  (one per ReAct round)
         │    ├── AGENT  invoke_agent          (PromptAgent.next_action)
         │    │    ├── TASK  build_prompt_context (PromptConstructor.construct)
         │    │    └── LLM  chat / text_completion
         │    │              * OpenAI provider — emitted by the OpenAI SDK probe
         │    │              * HuggingFace provider — emitted by THIS package
         │    └── TOOL  execute_tool {action_type}  (ScriptBrowserEnv.step)
         └── ...

    AGENT  create_agent                        (one-shot; construct_agent)

Design principles
-----------------

* **Do not double-emit OpenAI LLM spans.** WebArena's
  ``generate_from_openai_chat_completion`` / ``generate_from_openai_completion``
  ultimately call ``openai.ChatCompletion.create`` /
  ``openai.Completion.create`` which already have a dedicated OpenAI SDK
  instrumentor (e.g. ``opentelemetry-instrumentation-openai``). We rely on
  *that* instrumentor for token usage / model / finish-reason and let its
  LLM span attach itself naturally as a child of our AGENT span via the
  shared OTel context.
* **HuggingFace path is ours.** The ``text_generation`` client has no
  off-the-shelf probe, so we wrap
  ``llms.providers.hf_utils.generate_from_huggingface_completion`` to emit
  an LLM span for that path.
* **No invasive rewrite of ``run.py:test()``.** ENTRY / CHAIN / STEP are
  synthesised by latching on to ``ScriptBrowserEnv.reset`` (task start),
  ``ScriptBrowserEnv.close`` (batch end) and ``PromptAgent.next_action``
  (round start). See ``internal/_state.py`` for the state machine.

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.webarena import WebarenaInstrumentor

    WebarenaInstrumentor().instrument()

    # Then run WebArena as normal (e.g. ``python run.py ...``).
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.webarena.package import _instruments
from opentelemetry.instrumentation.webarena.version import __version__

logger = logging.getLogger(__name__)

__all__ = ["WebarenaInstrumentor"]


# WebArena uses *flat* package names (``setup.cfg`` declares ``packages =
# browser_env, agent, evaluation_harness, llms`` with no ``webarena.``
# prefix). Patch targets therefore use the bare module names.
_PATCH_TARGETS = (
    # (module, qualname, wrapper_attr_name)
    ("browser_env.envs", "ScriptBrowserEnv.reset", "_env_reset_wrapper"),
    ("browser_env.envs", "ScriptBrowserEnv.close", "_env_close_wrapper"),
    ("browser_env.envs", "ScriptBrowserEnv.step", "_env_step_wrapper"),
    ("agent.agent", "construct_agent", "_construct_agent_wrapper"),
    ("agent", "construct_agent", "_construct_agent_wrapper"),
    ("agent.agent", "PromptAgent.next_action", "_next_action_wrapper"),
)

# PromptConstructor.construct is abstract on the base class, so we patch
# the two known concrete subclasses individually.
_PROMPT_CONSTRUCTOR_TARGETS = (
    ("agent.prompts.prompt_constructor", "DirectPromptConstructor.construct"),
    ("agent.prompts.prompt_constructor", "CoTPromptConstructor.construct"),
)

_HF_TARGET = (
    "llms.providers.hf_utils",
    "generate_from_huggingface_completion",
)


class WebarenaInstrumentor(BaseInstrumentor):
    """An ``opentelemetry-instrumentation`` plugin for WebArena.

    Spans (see module docstring) are emitted via ``wrapt`` hooks on six
    framework functions plus an optional HuggingFace LLM hook. OpenAI LLM
    spans are intentionally **not** emitted here (the OpenAI SDK probe
    handles them).
    """

    _patched: list[tuple[str, str]] = []
    _patched_hf: bool = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__, __version__, tracer_provider=tracer_provider
        )

        from opentelemetry.instrumentation.webarena.internal._wrappers import (
            ConstructAgentWrapper,
            EnvCloseWrapper,
            EnvResetWrapper,
            EnvStepWrapper,
            HuggingFaceCompletionWrapper,
            NextActionWrapper,
            PromptConstructWrapper,
        )

        wrappers = {
            "_env_reset_wrapper": EnvResetWrapper(tracer),
            "_env_close_wrapper": EnvCloseWrapper(),
            "_env_step_wrapper": EnvStepWrapper(tracer),
            "_construct_agent_wrapper": ConstructAgentWrapper(tracer),
            "_next_action_wrapper": NextActionWrapper(tracer),
        }

        # --- core patches (mandatory) ------------------------------------
        type(self)._patched = []
        for module, qualname, wrapper_key in _PATCH_TARGETS:
            try:
                wrap_function_wrapper(
                    module=module,
                    name=qualname,
                    wrapper=wrappers[wrapper_key],
                )
                type(self)._patched.append((module, qualname))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "WebarenaInstrumentor: could not wrap %s.%s: %s",
                    module,
                    qualname,
                    exc,
                )

        # --- PromptConstructor (two concrete subclasses) ------------------
        prompt_wrapper = PromptConstructWrapper(tracer)
        for module, qualname in _PROMPT_CONSTRUCTOR_TARGETS:
            try:
                wrap_function_wrapper(
                    module=module, name=qualname, wrapper=prompt_wrapper
                )
                type(self)._patched.append((module, qualname))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "WebarenaInstrumentor: could not wrap %s.%s: %s",
                    module,
                    qualname,
                    exc,
                )

        # --- HuggingFace provider (optional, only if module imports OK) --
        try:
            wrap_function_wrapper(
                module=_HF_TARGET[0],
                name=_HF_TARGET[1],
                wrapper=HuggingFaceCompletionWrapper(tracer),
            )
            type(self)._patched_hf = True
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "WebarenaInstrumentor: skipping HuggingFace wrapper: %s", exc
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        from opentelemetry.instrumentation.webarena.internal import (
            _state as state,
        )

        # Always make sure we don't leak open spans on uninstrument.
        try:
            state.end_task_spans()
        except Exception:  # noqa: BLE001
            pass

        # Unwrap each successfully-patched target. We import the module
        # lazily so uninstrument doesn't fail when WebArena is no longer
        # importable (e.g. during teardown).
        for module, qualname in list(type(self)._patched):
            self._safe_unwrap(module, qualname)
        type(self)._patched = []

        if type(self)._patched_hf:
            self._safe_unwrap(_HF_TARGET[0], _HF_TARGET[1])
            type(self)._patched_hf = False

    @staticmethod
    def _safe_unwrap(module: str, qualname: str) -> None:
        try:
            import importlib  # noqa: PLC0415

            mod = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "WebarenaInstrumentor: could not import %s for unwrap: %s",
                module,
                exc,
            )
            return

        parts = qualname.split(".")
        try:
            target = mod
            for p in parts[:-1]:
                target = getattr(target, p)
            attr = getattr(target, parts[-1], None)
            if attr is not None and hasattr(attr, "__wrapped__"):
                setattr(target, parts[-1], attr.__wrapped__)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "WebarenaInstrumentor: could not unwrap %s.%s: %s",
                module,
                qualname,
                exc,
            )
