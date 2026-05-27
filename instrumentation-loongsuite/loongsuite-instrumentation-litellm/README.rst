LoongSuite LiteLLM Instrumentation
======================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-litellm.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-litellm/

This library provides automatic instrumentation for the
`LiteLLM <https://github.com/BerriAI/litellm>`_ library, which provides
a unified interface to 100+ LLM providers.

Installation
------------

::

     git clone https://github.com/alibaba/loongsuite-python-agent.git
     cd loongsuite-python-agent
     pip install ./instrumentation-loongsuite/loongsuite-instrumentation-litellm

Configuration
-------------

The instrumentation can be enabled/disabled using environment variables:

* ``ENABLE_LITELLM_INSTRUMENTOR``: Enable/disable instrumentation (default: true)

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
    import litellm

    # Instrument LiteLLM
    LiteLLMInstrumentor().instrument()

    # Use LiteLLM as normal
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )

Features
--------

This instrumentation automatically captures:

* LLM completion calls (sync and async)
* Streaming completions
* Embedding calls
* Retry mechanisms
* Tool/function calls
* Request and response metadata
* Token usage
* Model information

The instrumentation follows OpenTelemetry semantic conventions for GenAI operations.

References
----------

* `OpenTelemetry LiteLLM Instrumentation <https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/litellm/litellm.html>`_
* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `LiteLLM Documentation <https://docs.litellm.ai/>`_

