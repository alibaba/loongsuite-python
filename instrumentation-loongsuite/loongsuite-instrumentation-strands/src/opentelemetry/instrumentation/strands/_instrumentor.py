import logging
from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.strands.package import _instruments
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.strands._hooks import LoongsuiteHook

logger = logging.getLogger(__name__)


class StrandsInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        self._hook = LoongsuiteHook(handler)

        try:
            wrap_function_wrapper(
                module="strands.agent.agent",
                name="Agent.__init__",
                wrapper=self._agent_init_wrapper,
            )
        except Exception:
            logger.debug("Failed to wrap strands Agent.__init__", exc_info=True)

    def _agent_init_wrapper(self, wrapped, instance, args, kwargs):
        wrapped(*args, **kwargs)
        try:
            instance.add_hook(self._hook)
        except Exception:
            logger.debug("Failed to register loongsuite hook on Agent", exc_info=True)

    def _uninstrument(self, **kwargs):
        try:
            import strands.agent.agent as agent_module
            unwrap(agent_module.Agent, "__init__")
        except Exception:
            pass
