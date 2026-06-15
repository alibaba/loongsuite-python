# AGENTS.md

## Cursor Cloud specific instructions

This repo is the **OpenTelemetry Python Contrib / LoongSuite** monorepo: a collection of
Python instrumentation/exporter/propagator libraries (published as PyPI packages). There is
**no long-running application server**; "running the product" means exercising the
instrumentation via the test suites or a small instrumented script.

### Toolchain / how things are wired
- Package/dev manager is **`uv`** with a workspace (`[tool.uv.workspace]` in `pyproject.toml`).
  `uv sync` builds a single dev virtualenv at `.venv/` containing every workspace package plus
  the dev tools (`tox`, `tox-uv`, `pre-commit`). The update script runs `uv sync`.
- Test/lint orchestration is **`tox`**. Two configs exist:
  - `tox.ini` — upstream OTel packages (e.g. `tox -e py312-test-instrumentation-requests`).
  - `tox-loongsuite.ini` — LoongSuite packages; must pass `-c tox-loongsuite.ini`
    (e.g. `tox -c tox-loongsuite.ini -e py312-test-loongsuite-instrumentation-langchain-latest`).
- Prefer the venv binaries: `.venv/bin/tox`, `.venv/bin/python`. `uv` installs to
  `~/.local/bin` (already on PATH for the update script; add it to PATH in interactive shells
  if `uv` isn't found).

### Common commands
- Lint (runs pre-commit: ruff lint, ruff format, uv-lock, rstcheck): `.venv/bin/tox -e ruff`
- List envs: `.venv/bin/tox -l` / `.venv/bin/tox -c tox-loongsuite.ini -l`
- Run one package's tests: `.venv/bin/tox -e py312-test-instrumentation-<pkg>`
- `tox -e <pkg>` (no python prefix) and a bare `tox` run the full matrix across many Python
  versions — slow and needs interpreters that may not be installed. Scope to a `py312-...` env.

### Non-obvious gotchas
- **System build libs are required** for `uv sync` to compile native deps; they are installed
  outside the update script (already baked into the VM): `build-essential pkg-config libpq-dev
  default-libmysqlclient-dev librdkafka-dev libsnappy-dev python3-dev`. If `uv sync` fails with
  `pg_config not found`, `Python.h: No such file`, etc., re-install the matching `-dev` package.
- **Auto-instrumentation CLI (`opentelemetry-instrument`) currently breaks in the `uv sync`
  venv.** `pyproject.toml` pins `opentelemetry-api/sdk/semantic-conventions` to the upstream
  core **`main` branch** (bleeding edge), which is ahead of what the bundled logging
  instrumentor expects; the auto-instrumentation `sitecustomize` aborts with
  `LogRecord.__init__() got an unexpected keyword argument 'context'` and emits no telemetry.
  For a quick end-to-end smoke test, instrument **programmatically** instead (set up a
  `TracerProvider` + `ConsoleSpanExporter`, then `RequestsInstrumentor().instrument()`), which
  is also how the test suites run. The per-package `tox` envs are unaffected because they pin
  the core repo via `CORE_REPO_SHA`.
- `opentelemetry-instrument`/`opentelemetry-bootstrap` resolve the interpreter named `python`;
  the VM only has `python3`. Pass an explicit interpreter path if you use those CLIs.
- GenAI/LLM instrumentation tests use recorded **VCR cassettes**, so no live API keys are
  needed for the default suites.
- DB/broker integration tests under `tests/opentelemetry-docker-tests/` need the optional
  Docker Compose stack (Postgres/MySQL/Mongo/Redis/Jaeger) and are not run by default.
