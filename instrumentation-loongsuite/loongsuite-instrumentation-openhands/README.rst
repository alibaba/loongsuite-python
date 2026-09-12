OpenTelemetry OpenHands Instrumentation
========================================

Automatic OpenTelemetry instrumentation for the legacy OpenHands V0 /
CodeAct runtime.

What is covered
---------------

This package wraps the V0 ``python -m openhands.core.main`` execution path:

* ``openhands.core.main.run_controller`` for the ENTRY span.
* ``openhands.core.loop.run_agent_until_done`` for the AGENT span fallback.
* ``AgentController.__init__`` / ``AgentController.close`` for lifecycle-bound
  ENTRY and AGENT spans that survive ``python -m`` from-import binding.
* ``AgentController._step`` for ReAct STEP spans.
* ``Runtime.run_action`` for TOOL spans.
* ``LLM.__init__`` to bridge the current OpenHands context into LiteLLM calls.

Span tree
---------

::

    ENTRY  enter openhands
    `-- AGENT invoke_agent codeact
        |-- STEP  react step [xN]
        |   |-- LLM   chat {model}
        |   `-- TOOL  execute_tool {tool_name}
        `-- STEP  react step [...]

``python -m`` and from-import binding
-------------------------------------

When OpenHands V0 is launched via ``python -m openhands.core.main``, Python
executes ``main.py`` as ``__main__``. Symbols imported with ``from ... import``
can be bound before module-level wrappers are installed, so patching
``openhands.core.main.run_controller`` is not enough by itself.

To keep ENTRY and AGENT spans reliable, this instrumentation primarily opens
them from ``AgentController.__init__`` and closes them from
``AgentController.close``. The module-level wrappers remain as a fallback for
programmatic invocations.

Cross-thread context bridge
---------------------------

OpenHands V0 may execute controller steps and runtime tool calls in worker
threads with fresh asyncio loops. The instrumentation stores the active OTel
context by session id and re-attaches it in STEP, TOOL, and LLM bridge wrappers
so the trace remains:

``ENTRY -> AGENT -> STEP -> (LLM / TOOL)``.

Semantic-convention I/O capture
-------------------------------

STEP spans emit ``input.value`` / ``output.value`` alongside GenAI semantic
attributes. ENTRY and AGENT spans use only GenAI-native attributes — they do
not mirror OpenInference ``input.value`` / ``output.value``. TOOL spans never
set ``input.value`` / ``output.value``; they always set
``gen_ai.tool.call.arguments`` (JSON object string, ``"{}"`` when empty) and
``gen_ai.tool.call.result``.

* **ENTRY** emits ``gen_ai.input.messages`` and ``gen_ai.output.messages`` using
  the ARMS parts-based message schema.
* **AGENT** emits ``gen_ai.input.messages``, ``gen_ai.output.messages``,
  ``gen_ai.system_instructions``, and ``gen_ai.tool.definitions``.
* **STEP** emits recent input history and the pending assistant/tool-call
  output for the ReAct round.
* **TOOL** emits ``gen_ai.tool.name``, ``gen_ai.tool.type``,
  ``gen_ai.tool.call.id``, ``gen_ai.tool.description``,
  ``gen_ai.tool.call.arguments``, and ``gen_ai.tool.call.result``.

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.openhands import OpenHandsInstrumentor

    OpenHandsInstrumentor().instrument()

Configuration
-------------

Environment variables:

* ``OTEL_INSTRUMENTATION_OPENHANDS_ENABLED`` (default ``true``)
* ``OTEL_INSTRUMENTATION_OPENHANDS_OUTER_SPANS`` (default ``true``)
* ``OTEL_INSTRUMENTATION_OPENHANDS_AUTO_INSTRUMENT_LITELLM`` (default ``true``)

I/O capture is always on and content is emitted in full.
