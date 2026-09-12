# VitaBench DashScope Example

This example runs a single VitaBench delivery task with LoongSuite
instrumentation and DashScope's OpenAI-compatible chat completions endpoint.

Required environment variables:

```bash
export OPENAI_API_KEY=<your DashScope API key>
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```

Then run:

```bash
./setup.sh
./cmd.sh
```

`setup.sh` writes `models.yaml` with the full `/chat/completions` endpoint and
injects the API key via the `Authorization` header at runtime. Do not commit a
rendered `models.yaml` containing a real key.
