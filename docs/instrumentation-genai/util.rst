OpenTelemetry Python - GenAI Util
=================================

.. LoongSuite Extension: link the manual guide and expose the extended public API.

.. rstcheck: ignore-directives=autoclass,autofunction

For LoongSuite installation, runnable examples, field mappings, custom
attributes, and content/context configuration, see the
`LoongSuite manual instrumentation guide <https://github.com/alibaba/loongsuite-python/blob/main/util/opentelemetry-util-genai/README-loongsuite.rst>`_.

.. automodule:: opentelemetry.util.genai
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: opentelemetry.util.genai.utils
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: opentelemetry.util.genai.types
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: opentelemetry.util.genai.environment_variables
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: opentelemetry.util.genai.completion_hook
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: opentelemetry.util.genai._upload
    :members:
    :show-inheritance:

LoongSuite extended API
----------------------------------------

.. autoclass:: opentelemetry.util.genai.extended_handler.ExtendedTelemetryHandler
    :members:
    :show-inheritance:

.. autofunction:: opentelemetry.util.genai.extended_handler.get_extended_telemetry_handler

.. automodule:: opentelemetry.util.genai.extended_types
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: opentelemetry.util.genai.extended_memory
    :members: MemoryInvocation
    :show-inheritance:
