# LoongSuite VitaBench Instrumentation

OpenTelemetry instrumentation for the VitaBench multi-domain simulation framework.

## Installation

```bash
pip install loongsuite-instrumentation-vita
```

## Usage

```python
from opentelemetry.instrumentation.vita import VitaInstrumentor

VitaInstrumentor().instrument()
```
