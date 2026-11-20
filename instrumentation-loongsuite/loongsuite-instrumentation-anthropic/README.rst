LoongSuite Anthropic Instrumentation
====================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/loongsuite-instrumentation-anthropic.svg
   :target: https://pypi.org/project/loongsuite-instrumentation-anthropic/

This library allows tracing LLM requests made by the
`Anthropic Python SDK <https://pypi.org/project/anthropic/>`_ with the
LoongSuite distribution of the Anthropic instrumentation.

Installation
------------

::

    pip install ./util/opentelemetry-util-genai "./instrumentation-loongsuite/loongsuite-instrumentation-anthropic[instruments]"

If you don't have an Anthropic application yet, try our `examples <examples>`_
which only need a valid Anthropic API key.

Check out the `zero-code example <examples/zero-code>`_ for a quick start.

Usage
-----

This section describes how to set up Anthropic instrumentation if you're setting OpenTelemetry up manually.
Check out the `manual example <examples/manual>`_ for more details.

.. code-block:: python

    from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
    import anthropic

    # Instrument Anthropic
    AnthropicInstrumentor().instrument()

    # Use Anthropic client as normal
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, Claude!"}
        ]
    )


Configuration
-------------

Capture Message Content
***********************

By default, prompts and completions are not captured. To enable message content capture,
set both environment variables:

::

    export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY


References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `Anthropic Documentation <https://docs.anthropic.com/>`_
