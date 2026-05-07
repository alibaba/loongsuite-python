"""Tests for WildToolInstrumentor lifecycle."""

from opentelemetry.instrumentation.wildtool import WildToolInstrumentor


class TestWildToolInstrumentor:
    def test_instrument_and_uninstrument(self, tracer_provider):
        instrumentor = WildToolInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True,
        )
        assert instrumentor._handler is not None
        instrumentor.uninstrument()
        assert instrumentor._handler is None

    def test_instrumentation_dependencies(self):
        instrumentor = WildToolInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert ("openai >= 1.0.0",) == deps
