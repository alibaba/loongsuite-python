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

"""LLM async-compat wrapper for Cognee ``GenericAPIAdapter``.

Background
----------
Cognee v1.2.1 constructs its LLM client via::

    self.aclient = instructor.from_litellm(litellm.acompletion, mode=...)

When ``loongsuite-instrumentation-litellm`` is loaded (which the Cognee
README mandates), it patches ``litellm.acompletion`` to an instance of
``opentelemetry.instrumentation.litellm._wrapper.AsyncCompletionWrapper``
— a class instance with an ``async def __call__``. Instructor 1.14.5's
``instructor.utils.core.is_async`` uses ``inspect.iscoroutinefunction``,
which returns ``False`` for callable class instances regardless of
whether ``__call__`` is ``async def``. Instructor therefore picks the
sync ``Instructor`` + ``new_create_sync`` retry path; ``retry_sync``
calls ``litellm.acompletion(...)`` *without* ``await`` and the resulting
coroutine is mistaken for the API response:

    instructor.core.exceptions.InstructorRetryException:
      'coroutine' object has no attribute 'choices'

Fix
---
After ``GenericAPIAdapter.__init__`` runs, this wrapper rebuilds
``self.aclient`` as an explicit ``AsyncInstructor`` by routing
``litellm.acompletion`` through a real ``async def`` (so
``iscoroutinefunction`` returns ``True`` and instructor picks
``new_create_async``). All retry / fallback / content-policy logic in
``GenericAPIAdapter.acreate_structured_output`` is preserved unchanged.

Semantics
---------
The wrap is **transparent** to telemetry:

* It does not change ``response_model`` / ``mode`` / ``max_retries`` /
  ``llm_args`` handling — those still flow through Cognee's original
  ``acreate_structured_output``.
* It does not wrap ``litellm.acompletion`` itself; ``litellm.acompletion``
  is still the LiteLLM-instrumented callable, so LLM spans + token usage
  metrics are still produced by ``loongsuite-instrumentation-litellm``.
* It does not create any new span / metric — the Cognee instrumentor's
  span coverage (ENTRY/AGENT/TOOL/STEP/EMBEDDING) is unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)


_ADAPTER_MODULE = (
    "cognee.infrastructure.llm.structured_output_framework."
    "litellm_instructor.llm.generic_llm_api.adapter"
)
_ADAPTER_CLASS = "GenericAPIAdapter"


def _build_async_acompletion(original_acompletion: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``litellm.acompletion`` in a real ``async def`` so instructor
    detects it as async via ``inspect.iscoroutinefunction``.

    The wrapper resolves ``litellm.acompletion`` at call time so any
    re-instrumentation of the global (e.g., LiteLLMInstrumentor
    re-install) is picked up automatically.
    """

    async def _async_acompletion(*args: Any, **kwargs: Any) -> Any:
        import litellm  # late import — resolve the *current* global

        return await litellm.acompletion(*args, **kwargs)

    # Preserve functools.WRAPPER_ASSIGNMENTS for debuggability.
    try:
        _async_acompletion.__name__ = "acompletion"
        _async_acompletion.__doc__ = original_acompletion.__doc__
    except AttributeError:
        pass

    return _async_acompletion


def _rebuild_aclient_as_async(instance: Any) -> bool:
    """Rebuild ``instance.aclient`` as an ``AsyncInstructor``.

    Returns ``True`` if rebuilt, ``False`` if already async or rebuild skipped.
    """
    aclient = getattr(instance, "aclient", None)
    if aclient is None:
        return False

    cls_name = type(aclient).__name__
    if cls_name in ("AsyncInstructor", "_AsyncInstructor"):
        return False

    try:
        import instructor  # type: ignore
        import litellm  # type: ignore
    except ImportError:
        logger.debug(
            "Cannot rebuild aclient: instructor or litellm not importable"
        )
        return False

    mode = getattr(aclient, "mode", None) or instructor.Mode("json_mode")
    async_acompletion = _build_async_acompletion(litellm.acompletion)
    try:
        instance.aclient = instructor.from_litellm(async_acompletion, mode=mode)
        logger.debug(
            "Rebuilt GenericAPIAdapter.aclient as AsyncInstructor "
            "(was %s, mode=%s)",
            cls_name,
            mode,
        )
        return True
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Failed to rebuild aclient as AsyncInstructor: %s", e)
        return False


def _make_init_wrapper() -> Callable[..., Any]:
    def _init_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        result = wrapped(*args, **kwargs)
        try:
            _rebuild_aclient_as_async(instance)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(
                "GenericAPIAdapter.aclient async rebuild failed: %s", e
            )
        return result

    return _init_wrapper


def install_llm_compat_wrapper() -> None:
    """Wrap ``GenericAPIAdapter.__init__`` so ``self.aclient`` is async-safe."""
    try:
        import importlib

        module = importlib.import_module(_ADAPTER_MODULE)
        cls = getattr(module, _ADAPTER_CLASS)
        wrap_function_wrapper(cls, "__init__", _make_init_wrapper())
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(
            "Failed to wrap %s.%s.__init__: %s",
            _ADAPTER_MODULE,
            _ADAPTER_CLASS,
            e,
        )


def uninstall_llm_compat_wrapper() -> None:
    from opentelemetry.instrumentation.utils import unwrap

    try:
        import importlib

        module = importlib.import_module(_ADAPTER_MODULE)
        cls = getattr(module, _ADAPTER_CLASS)
        unwrap(cls, "__init__")
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(
            "Failed to unwrap %s.%s.__init__: %s",
            _ADAPTER_MODULE,
            _ADAPTER_CLASS,
            e,
        )
