LoongSuite Google GenAI SDK Instrumentation
===========================================

This package aligns with
``opentelemetry-instrumentation-google-genai==1.0b1`` from the canonical
``open-telemetry/opentelemetry-python-genai`` repository, but it does not use
the upstream 1.x GenAI util at runtime. Provider hooks are adapted to
LoongSuite's own ``opentelemetry-util-genai`` source package and
``ExtendedTelemetryHandler``. LoongSuite release packaging renames that shared
util to ``loongsuite-otel-util-genai`` and replaces the renamed community
Google GenAI package with this implementation.

Supported operations
--------------------

* synchronous and asynchronous ``generate_content``;
* synchronous and asynchronous ``generate_content_stream``;
* synchronous and asynchronous ``embed_content``;
* synchronous and asynchronous ``interactions.create``, including streams;
* SDK automatic function calling with nested ``execute_tool`` spans;
* Gemini Developer API and Vertex AI provider identification;
* text, reasoning, inline binary, URI, function-call, and function-response
  message parts;
* request parameters, tool definitions, response IDs/models, token usage,
  finish reasons, streaming TTFT, and ``hook_advice`` fail-open error handling;
* opt-in Google-specific request configuration attributes through
  ``OTEL_GOOGLE_GENAI_GENERATE_CONTENT_CONFIG_INCLUDES`` and
  ``OTEL_GOOGLE_GENAI_GENERATE_CONTENT_CONFIG_EXCLUDES``.

The Interactions API is available in recent ``google-genai`` 2.x releases.
Generation and embedding remain supported from ``google-genai`` 1.32 onward.
Instrumentation-only failures are degraded to missing telemetry: the original
SDK call or application tool executes once, and its return object, streamed
chunks, cancellation, ``GeneratorExit``, and exception identity are preserved.

Installation
------------

::

    pip install loongsuite-instrumentation-google-genai google-genai

Usage
-----

::

    from google import genai
    from opentelemetry.instrumentation.google_genai import (
        GoogleGenAiSdkInstrumentor,
    )

    GoogleGenAiSdkInstrumentor().instrument()
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Explain OpenTelemetry in one sentence.",
    )

Message content follows the shared GenAI util content-capture configuration.
For example, use
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY`` to record
messages on spans.

Metrics and commercial extensions
---------------------------------

Span lifecycle, standard ``gen_ai.client.*`` metrics, multimodal processing,
and completion hooks are owned by LoongSuite's shared GenAI util rather than
implemented again in this provider plugin.

The Robin commercial package keeps the same provider hooks but supplies its
commercial GenAI util underneath ``ExtendedTelemetryHandler``. That util adds
ARMS metrics and other enterprise behavior. Robin also extends the
provider-level suppression check to honor its private
``_SUPPRESS_LLM_SDK_KEY`` so a high-level framework span can suppress only the
nested Google GenAI SDK spans. The private key is intentionally not part of
this OSS package.

Upstream synchronization
------------------------

The upstream baseline and the intentionally retained LoongSuite delta are
documented in ``UPSTREAM.md``. Provider hook files preserve the upstream
layout, while ``_compat.py`` contains the removable adapter to LoongSuite's
extended GenAI util.

Live verification
-----------------

The regular test matrix does not require external credentials. To exercise the
latest supported SDK against the real Gemini Developer API, provide your own
restricted key through the environment and opt in explicitly:

::

    RUN_GOOGLE_GENAI_LIVE_TESTS=1 \
      tox -c tox-loongsuite.ini \
      -e py312-test-loongsuite-instrumentation-google-genai-latest \
      -- -k live_google_genai

The local-only live test reads ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) and
uses ``gemini-3.5-flash-lite`` by default. Set the local
``GOOGLE_GENAI_LIVE_MODEL`` environment variable to change the model. Public CI
only replays redacted VCR cassettes and does not require provider credentials.
