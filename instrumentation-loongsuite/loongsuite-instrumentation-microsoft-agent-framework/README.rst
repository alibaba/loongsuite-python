========================================================
LoongSuite Microsoft Agent Framework Instrumentation
========================================================

This package provides LoongSuite instrumentation for
`Microsoft Agent Framework <https://github.com/microsoft/agent-framework>`_
(``agent-framework-core``).

It implements the hybrid "SpanProcessor + optional ReAct step patch" plan
described in ``llm-dev/microsoft-agent-framework/investigate/execute.md``:

* ``MAFSemanticProcessor`` enriches the native OTel spans emitted by MAF's
  ``ChatTelemetryLayer`` / ``EmbeddingTelemetryLayer`` /
  ``AgentTelemetryLayer`` / workflow helpers with the ARMS GenAI semantic
  conventions (``gen_ai.span.kind``, ``gen_ai.operation.name``, normalized
  attribute names, ``gen_ai.response.time_to_first_token``, etc.) and
  aggregates the 6 ARMS gauges. Successful spans keep the OpenTelemetry
  default ``UNSET`` status; failed spans keep MAF's ``ERROR`` status.
* ``react_step_patch`` (opt-in via ``ARMS_MAF_REACT_STEP_ENABLED=true``)
  wraps ``FunctionInvocationLayer.get_response`` so that each LLM round-trip
  inside the ReAct loop emits one ``react step`` span via
  ``ExtendedTelemetryHandler.react_step()`` from ``opentelemetry-util-genai``.

Truncation / PII helpers are reused from ``opentelemetry.util.genai.utils``
(``gen_ai_json_dumps``), aligned with
the OpenAI Agents v2 GenAI instrumentation.

Installation
============

Install the instrumentation package together with Microsoft Agent Framework in
the target application environment:

.. code-block:: console

   pip install "loongsuite-instrumentation-microsoft-agent-framework[instruments]"

To keep the framework dependency controlled by the application, install the
instrumentation package and framework separately:

.. code-block:: console

   pip install loongsuite-instrumentation-microsoft-agent-framework
   pip install agent-framework-core

Configuration
=============

======================================  ==========  ==============================================================
Env                                     Default     Description
======================================  ==========  ==============================================================
``ARMS_MAF_INSTRUMENTATION_ENABLED``    ``true``    Master switch; ``false`` disables instrumentation.
``ARMS_MAF_SENSITIVE_DATA_ENABLED``     ``false``   Capture inputs/outputs (linked to MAF's sensitive-data option).
``ARMS_MAF_REACT_STEP_ENABLED``         ``false``   Emit ``react step`` spans for non-streaming ReAct tool loops.
``ARMS_MAF_METRICS_ENABLED``            ``true``    Aggregate ARMS GenAI gauges in-process.
``ARMS_MAF_SLOW_THRESHOLD_MS``          ``1000``    Slow-call threshold in ms.
======================================  ==========  ==============================================================
