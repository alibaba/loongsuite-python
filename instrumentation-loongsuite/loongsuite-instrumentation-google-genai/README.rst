LoongSuite Google GenAI SDK Instrumentation
===========================================

This package instruments ``google-genai`` 1.x and 2.x and routes telemetry
through LoongSuite's ``opentelemetry-util-genai`` extension. It replaces the
renamed community Google GenAI package in LoongSuite release builds.

Supported operations
--------------------

* synchronous and asynchronous ``generate_content``;
* synchronous and asynchronous ``generate_content_stream``;
* synchronous and asynchronous ``embed_content``;
* Gemini Developer API and Vertex AI provider identification;
* text, reasoning, inline binary, URI, function-call, and function-response
  message parts;
* request parameters, tool definitions, response IDs/models, token usage,
  finish reasons, streaming TTFT, and fail-open error handling.

The SDK's experimental Interactions API and cache-management APIs are not
instrumented by this package. The supported generation and embedding surface
is shared by ``google-genai`` 1.x and 2.x.

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
For example, use ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental``
and ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY`` to record
messages on spans.

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
