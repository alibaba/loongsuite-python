# LoongSuite Instrumentation Common

Shared failure-isolation primitives for LoongSuite Python instrumentations.

The package keeps application execution outside instrumentation error boundaries:

- `hook_advice` and `async_hook_advice` protect instrumentation-only callbacks.
- `IsolatedStream` and `IsolatedAsyncStream` preserve the wrapped stream protocol
  while treating chunk processing and finalization callbacks as best effort.
- Application return values, chunks, exceptions, and context-manager suppression
  results are never replaced by ordinary instrumentation exceptions.

Decorators must not be applied to a function containing the application call or
to generator and async-generator functions. Stream lifecycle instrumentation
must use the stream proxies instead.

```python
from opentelemetry.instrumentation.loongsuite import (
    IsolatedStream,
    hook_advice,
)


@hook_advice("example", "prepare")
def prepare_advice(request):
    return build_invocation(request)


def on_chunk(chunk):
    record_chunk_attributes(chunk)


def on_finish():
    finish_span()


invocation = prepare_advice(request)
response = call_application_once()
return IsolatedStream(
    response,
    on_chunk=on_chunk,
    on_finish=on_finish,
)
```

Stream callbacks are passed undecorated because the proxy invokes them through
`call_advice`; pre-decorating them would create duplicate commercial
self-monitoring events after synchronization.

Commercial distributions can retain the stream implementation and adapt only
the public advice decorators to their self-monitoring implementation.
