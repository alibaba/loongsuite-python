==============================
Microsoft Agent Framework Instrumentation
==============================

This package provides OpenTelemetry instrumentation for
`Microsoft Agent Framework <https://github.com/microsoft/agent-framework>`_
(``agent-framework-core``).

It implements the hybrid "SpanProcessor + optional ReAct step patch" plan
described in ``llm-dev/microsoft-agent-framework/investigate/execute.md``:

* ``MAFSemanticProcessor`` enriches the native OTel spans emitted by MAF's
  ``ChatTelemetryLayer`` / ``EmbeddingTelemetryLayer`` /
  ``AgentTelemetryLayer`` / workflow helpers with the ARMS GenAI semantic
  conventions (``gen_ai.span.kind``, ``gen_ai.operation.name``, normalized
  attribute names, ``gen_ai.response.time_to_first_token``, ``StatusCode.OK``
  on success, etc.) and aggregates the 6 ARMS gauges.
* ``react_step_patch`` (opt-in via ``ARMS_MAF_REACT_STEP_ENABLED=true``)
  wraps ``FunctionInvocationLayer.get_response`` so that each LLM round-trip
  inside the ReAct loop emits one ``react step`` span via
  ``ExtendedTelemetryHandler.react_step()`` from ``opentelemetry-util-genai``.

Truncation / PII helpers are reused from ``opentelemetry.util.genai.utils``
(``gen_ai_json_dumps``), aligned with
``instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/.../span_processor.py``.

Configuration
============

============================  ==========  ==========================================
Env                           Default     Description
============================  ==========  ==========================================
``ARMS_MAF_INSTRUMENTATION_ENABLED``      ``true``    Master switch; ``false`` disables instrumentation.
``ARMS_MAF_SENSITIVE_DATA_ENABLED``      ``false``   Capture inputs/outputs (linked to MAF's ``ENABLE_SENSITIVE_DATA``).
``ARMS_MAF_REACT_STEP_ENABLED``          ``false``   Emit ``react step`` spans (opt-in).
``ARMS_MAF_SLOW_THRESHOLD_MS``           ``1000``    Slow-call threshold in ms.
============================  ==========  ==========================================
