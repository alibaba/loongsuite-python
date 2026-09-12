# LoongSuite WideSearch Instrumentation

OpenTelemetry instrumentation for the [WideSearch](https://github.com/ByteDance-Seed/WideSearch) multi-agent search framework.

## Installation

```bash
pip install loongsuite-instrumentation-widesearch
```

## Usage

```python
from opentelemetry.instrumentation.widesearch import WideSearchInstrumentor

WideSearchInstrumentor().instrument()
```
