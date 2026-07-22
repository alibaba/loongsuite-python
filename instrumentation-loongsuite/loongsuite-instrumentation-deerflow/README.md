# LoongSuite DeerFlow Instrumentation

LoongSuite instrumentation for [DeerFlow](https://github.com/bytedance/deer-flow) 2.x.

## Features

- Gateway and embedded-client `ENTRY` spans
- Lead-agent and subagent `AGENT` spans
- One `STEP` span for every direct model-decision node
- Existing LangChain `LLM`, `TOOL`, and supporting `CHAIN` spans
- Session, user, assistant, run, and DeerFlow correlation identities
- Streaming, cancellation, timeout, error, and concurrent-run lifecycle handling
- Coexistence with DeerFlow's LangSmith, Langfuse, and Monocle callbacks

The expected core tree is:

```text
ENTRY deerflow
└── AGENT lead-agent
    └── STEP
        ├── LLM
        └── TOOL task
            └── AGENT subagent:<name>
                └── STEP → LLM / TOOL
```

Direct invocation of a graph returned by `create_deerflow_agent()` starts at
the `AGENT` span. It does not create a synthetic application `ENTRY` span.

## Compatibility

- DeerFlow `>=2,<3`
- Python 3.12+
- DeerFlow installed from the official source repository

The similarly named `deerflow-harness` package published on PyPI is not the
official DeerFlow 2.x distribution. This instrumentation deliberately does not
offer an `instruments` extra that could install that placeholder package.

## Choose an integration mode

DeerFlow has several official startup modes. The instrumentation must be
installed in the Python environment used by the Gateway, not in the frontend,
nginx, sandbox, or an unrelated system Python environment.

| DeerFlow mode | Recommended LoongSuite integration | Application source change |
| --- | --- | --- |
| Local foreground or daemon | `loongsuite-instrument` around `scripts/serve.sh` | No |
| Docker development or production | Lock `loongsuite-site-bootstrap` and the instrumentation into the backend uv project | No Python source change |
| Embedded `DeerFlowClient` or direct SDK graph | Programmatic `DeerFlowInstrumentor` | One initialization call |

## Local foreground and daemon

### 1. Prepare DeerFlow first

Follow DeerFlow's official setup and let it create `backend/.venv` before
installing LoongSuite:

```bash
git clone https://github.com/bytedance/deer-flow.git
cd deer-flow
make setup
make install
```

### 2. Install LoongSuite into the Gateway environment

```bash
uv pip install \
  --python backend/.venv/bin/python \
  loongsuite-distro \
  loongsuite-instrumentation-deerflow
```

Install an exporter as well when it is not already present. For OTLP:

```bash
uv pip install \
  --python backend/.venv/bin/python \
  opentelemetry-exporter-otlp
```

For development against a LoongSuite source checkout, follow the main
[source-install instructions](../../README.md#install-from-source-for-development)
and install this package from its local directory into the same
`backend/.venv`.

### 3. Verify locally with the console exporter

Run DeerFlow through the executable from `backend/.venv`:

```bash
backend/.venv/bin/loongsuite-instrument \
  --traces_exporter console \
  --metrics_exporter none \
  --service_name deerflow \
  ./scripts/serve.sh --dev --skip-install
```

The instrumentation bootstrap propagates through `serve.sh` into the Gateway
Python process. The frontend and nginx processes continue to start normally.

For a local production daemon, configure the exporter as shown below and run:

```bash
backend/.venv/bin/loongsuite-instrument \
  ./scripts/serve.sh --prod --daemon --skip-install
```

> **Why `--skip-install` is required:** DeerFlow's normal `make dev` and
> `make start` paths run `uv sync`. An exact `uv sync` removes packages that
> are not recorded in DeerFlow's `pyproject.toml` and `uv.lock`. Install
> LoongSuite after the last sync and use `--skip-install`, or use the persistent
> uv-locked approach below. A subsequent `uv run` preserves the installed
> packages; a subsequent `uv sync` does not.

## Persistent local and Docker integration

DeerFlow Docker development runs `uv sync --all-packages` every time the
Gateway container starts. Production images also construct the backend virtual
environment from the uv lock. A host-side `pip install` therefore does not
instrument either container and an image-only ad-hoc install is not durable for
Docker development.

For an integration that survives all official startup modes, add the
LoongSuite packages to DeerFlow's backend project:

```bash
cd backend
uv add \
  'loongsuite-distro[otlp]' \
  loongsuite-site-bootstrap \
  loongsuite-instrumentation-deerflow
cd ..
```

This intentionally updates DeerFlow's `backend/pyproject.toml` and `uv.lock`.
Commit those deployment dependency changes in your DeerFlow deployment fork so
that local development, Docker development, and production image builds use the
same packages.

Add the bootstrap and exporter settings to DeerFlow's root `.env` file:

```bash
LOONGSUITE_PYTHON_SITE_BOOTSTRAP=True
LOONGSUITE_PYTHON_SITE_BOOTSTRAP_LOG_SUCCESS=False

OTEL_SERVICE_NAME=deerflow
OTEL_TRACES_EXPORTER=otlp
OTEL_METRICS_EXPORTER=otlp
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
```

Do not commit credentials or authentication headers from exporter settings.
DeerFlow's local launcher and Docker Compose files load the root `.env`. The
site bootstrap initializes LoongSuite before DeerFlow imports its Gateway,
agent factory, or cached `create_agent` aliases. No change to
`backend/app/gateway/app.py` is required.

You can then keep the official commands:

```bash
# Local development
make dev

# Docker development
make docker-start

# Docker production
make up
```

For a backend outside the Compose network, replace the example collector host
with a reachable OTLP endpoint. For OTLP/HTTP use `http/protobuf` and the
corresponding endpoint instead.

### Container verification

Verify that the three required packages are installed in the Gateway
environment:

```bash
docker exec deer-flow-gateway \
  /app/backend/.venv/bin/python -c \
  'from importlib.metadata import version; print(version("loongsuite-distro")); print(version("loongsuite-site-bootstrap")); print(version("loongsuite-instrumentation-deerflow"))'
```

Then execute a DeerFlow request and verify that the exporter receives one
`ENTRY` and one lead `AGENT`, with one `STEP` for each model decision. A tool
loop should report model finish reasons in the order `tool_calls`, then `stop`.

## Programmatic instrumentation

Use programmatic setup for an embedded client or an application that already
owns its OpenTelemetry SDK initialization. Configure a tracer provider and
exporter first, then instrument before importing DeerFlow application modules:

```python
from opentelemetry.instrumentation.deerflow import DeerFlowInstrumentor

DeerFlowInstrumentor().instrument()

from deerflow.client import DeerFlowClient

client = DeerFlowClient()
```

Call `DeerFlowInstrumentor().uninstrument()` only during controlled teardown,
such as tests. The instrumentor composes the LoongSuite LangChain and LangGraph
instrumentations when they have not already been enabled.

## Configuration

### OTLP export

```bash
export OTEL_SERVICE_NAME=deerflow
export OTEL_TRACES_EXPORTER=otlp
export OTEL_METRICS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317
```

Authentication headers, TLS certificates, and signal-specific endpoints use
the standard OpenTelemetry exporter environment variables.

### Content capture

Message content is disabled by default. To capture it explicitly:

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```

Supported capture modes follow the LoongSuite GenAI configuration. Keep the
default no-content mode for environments where prompts, tool arguments, or
responses can contain sensitive data.

## Span and identity behavior

- An existing host `ENTRY` takes precedence over DeerFlow Gateway and embedded
  client entry points.
- `DeerFlowClient.chat()` reuses `stream()` and creates only one `ENTRY`.
- The embedded iterator restores its call-time context around each iterator
  advance and close, including cross-thread consumption.
- `gen_ai.session.id` and `gen_ai.user.id` prefer existing baggage values over
  DeerFlow identities.
- DeerFlow's request trace ID is recorded as `deerflow.trace.id`. It is a
  correlation field and never replaces the W3C OpenTelemetry trace ID.
- Gateway runs record `deerflow.run.id`. The public embedded stream API does
  not expose its internal run ID, so embedded spans omit that attribute.
- `task` remains a normal LangChain `TOOL`. No duplicate `TASK` span is added.
- Sandbox and skill work remains visible through downstream tool and model
  calls. This package does not add separate sandbox-allocation, skill-lifecycle,
  or background-memory spans.

## Troubleshooting

### The instrumentor silently does nothing

Confirm that the Gateway interpreter sees the official distribution and a
supported version:

```bash
backend/.venv/bin/python -c \
  'import deerflow; from importlib.metadata import version; print(version("deerflow-harness"))'
```

The version must be `>=2,<3`. A `deerflow` module without matching
`deerflow-harness` distribution metadata is intentionally ignored.

### Traces disappear after restarting DeerFlow

The most common cause is a later `uv sync` removing an ad-hoc instrumentation
install. Either reinstall LoongSuite after the sync and start with
`--skip-install`, or record the packages in DeerFlow's uv project and use
`loongsuite-site-bootstrap`.

### Docker exports no data

Check all three boundaries separately:

1. The packages exist inside `/app/backend/.venv`, not only on the host.
2. `LOONGSUITE_PYTHON_SITE_BOOTSTRAP=True` is present in the Gateway container.
3. The OTLP endpoint is reachable from the Gateway container; `localhost`
   refers to that container, not the host or a separate collector container.

### A direct SDK call has no `ENTRY`

This is expected. Direct `create_deerflow_agent(...).invoke()` or `.stream()`
starts at `AGENT`. Use the Gateway or `DeerFlowClient` when an application
`ENTRY` lifecycle is required, or provide a host `ENTRY` in the embedding
application.

## License

Apache License 2.0
