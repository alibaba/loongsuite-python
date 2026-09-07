LoongSuite Pydantic AI Instrumentation
======================================

This package adapts Pydantic AI's built-in OpenTelemetry Instrumentation
Capability to LoongSuite GenAI semantic conventions.

Use ``PydanticAIInstrumentor().instrument()`` to enable Pydantic AI's global
agent and embedding instrumentation defaults and register the LoongSuite span
processor. Add ``LoongSuiteInstrumentationCapability`` to agents that need
ReAct STEP spans:

.. code-block:: python

    from opentelemetry.instrumentation.pydantic_ai import (
        LoongSuiteInstrumentationCapability,
        PydanticAIInstrumentor,
    )
    from pydantic_ai import Agent

    PydanticAIInstrumentor().instrument()
    agent = Agent(
        "openai:gpt-4o-mini",
        capabilities=[LoongSuiteInstrumentationCapability()],
    )
