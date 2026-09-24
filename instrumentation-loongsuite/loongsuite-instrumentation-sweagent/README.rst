LoongSuite instrumentation for SWE-agent
========================================

This package instruments `SWE-agent <https://github.com/SWE-agent/SWE-agent>`_
using ``opentelemetry-util-genai`` ``ExtendedTelemetryHandler`` so traces align
with other LoongSuite GenAI plugins.

Spans
-----

.. list-table::
   :header-rows: 1

   * - SWE-agent locus
     - Span name (typical)
     - ``gen_ai.operation.name`` / ``gen_ai.span.kind``

   * - ``CombinedRunHooks.on_instance_start`` → ``on_instance_completed``
     - ``enter_ai_application_system``
     - ``enter`` / ``ENTRY``

   * - ``CombinedAgentHook.on_run_start`` → ``on_run_done``
     - ``invoke_agent swe-agent`` (``gen_ai.agent.name``); ``input_messages`` / ``output_messages`` match the entry span (same problem-statement user text via thread-local from ``on_instance_start``, same run summary as ``AgentRunResult`` via ``info`` + ``trajectory``); ``gen_ai.conversation.id`` = problem id when set; token usage from ``info.model_stats`` when present
     - ``invoke_agent`` / ``AGENT``

   * - ``CombinedAgentHook.on_step_start`` → ``on_step_done``
     - ``react step``
     - ``react`` / ``STEP``

   * - ``DefaultAgent.handle_action`` (bash / ``communicate``)
     - ``execute_tool <name>`` where ``name`` is the first LLM ``tool_calls[*].function.name`` when ``StepOutput.tool_calls`` is set (function-calling mode); otherwise ``execute_tool sweagent_bash`` (thought/action parsing). Tool arguments recorded as ``function.arguments`` (JSON decoded when possible), else ``step.action``.
     - ``execute_tool`` / ``TOOL``

Remote LLM calls (LiteLLM) are **not** duplicated here; enable
``loongsuite-instrumentation-litellm`` (or equivalent) for model spans.

Requirements
------------

- Python **3.11+** (matches upstream SWE-agent).

Installation
------------

From the LoongSuite repo root (after installing ``sweagent`` and ``opentelemetry-util-genai``):

::

    pip install -e ./util/opentelemetry-util-genai
    pip install -e ./instrumentation-loongsuite/loongsuite-instrumentation-sweagent

Usage
-----

::

    from opentelemetry.instrumentation.sweagent import SweagentInstrumentor
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    # add_span_processor(...)  # e.g. OTLP or console

    SweagentInstrumentor().instrument(tracer_provider=provider)
    # ... run sweagent ...
    SweagentInstrumentor().uninstrument()

Entry span input is derived from ``problem_statement.id`` and a truncated
``get_problem_statement()`` body. Output summarizes ``AgentRunResult.info`` and
trajectory length. Tool span arguments/results follow GenAI content-capture
environment variables when experimental semconv is enabled.
