# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scope the owning agent name to QwenPaw's background Dream coroutine."""

from importlib import import_module

from wrapt import wrap_function_wrapper

from opentelemetry import baggage, context
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai import hook_advice
from opentelemetry.util.genai.handler import _safe_detach

_MODULE = "qwenpaw.agents.memory.reme_light_memory_manager"


@hook_advice("qwenpaw", "attach_dream_owner")
def _attach_owner(instance):
    # Resolve per invocation, matching normal QwenPaw agent naming and reloads.
    config = import_module("qwenpaw.config.config")
    name = config.load_agent_config(instance.agent_id).name or "QwenPaw"
    if not isinstance(name, str) or not name.strip():
        return None
    owner_context = baggage.set_baggage("gen_ai.agent.name", name)
    # ReMe may create a helper Agent named "default" inside this scope.
    # Keep the owner override local; it is not an exported span attribute.
    owner_context = context.set_value(
        "qwenpaw.dream.agent.name", name, owner_context
    )
    return context.attach(owner_context)


async def _dream_wrapper(wrapped, instance, args, kwargs):
    token = _attach_owner(instance)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _detach_owner(token)


@hook_advice("qwenpaw", "detach_dream_owner")
def _detach_owner(token):
    _safe_detach(token)


@hook_advice("qwenpaw", "instrument_dream")
def instrument_dream():
    # Optional backend: absence must not prevent normal Entry instrumentation.
    try:
        module = import_module(_MODULE)
        cls = module.ReMeLightMemoryManager
    except (ImportError, AttributeError):
        return None
    if not callable(getattr(cls, "dream", None)):
        return None
    wrap_function_wrapper(cls, "dream", _dream_wrapper)
    return cls


@hook_advice("qwenpaw", "uninstrument_dream")
def uninstrument_dream(cls):
    if cls is not None:
        unwrap(cls, "dream")
