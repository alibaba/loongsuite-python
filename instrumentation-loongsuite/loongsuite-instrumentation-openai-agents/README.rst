OpenTelemetry OpenAI Agents SDK Instrumentation
================================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/loongsuite-instrumentation-openai-agents.svg
   :target: https://pypi.org/project/loongsuite-instrumentation-openai-agents/

This library provides automatic instrumentation for the
`OpenAI Agents SDK <https://github.com/openai/openai-agents-python>`_,
capturing telemetry data for agent runs, tool executions, LLM generations,
handoffs, and guardrails.

Installation
------------

::

    pip install loongsuite-instrumentation-openai-agents

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor

    OpenAIAgentsInstrumentor().instrument()

    # Your OpenAI Agents SDK code works as normal
    from agents import Agent, Runner

    agent = Agent(name="assistant", instructions="You are helpful.")
    result = Runner.run_sync(agent, "Hello!")

The instrumentation automatically captures:

- Agent invocation spans (``invoke_agent``)
- Tool execution spans (``execute_tool``)
- LLM generation spans (``chat``)
- Agent handoff spans
- Guardrail execution spans

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenAI Agents SDK documentation <https://openai.github.io/openai-agents-python/>`_
* `OpenTelemetry GenAI Semantic Conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
