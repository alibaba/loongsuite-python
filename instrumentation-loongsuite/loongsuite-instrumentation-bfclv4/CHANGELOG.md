# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial release of `loongsuite-instrumentation-bfclv4`.
- ENTRY span around `bfcl_eval._llm_response_generation.generate_results`.
- AGENT span around `bfcl_eval.model_handler.base_handler.BaseHandler.inference`
  with cross-thread OTel context propagation via a narrow patch of
  `bfcl_eval._llm_response_generation.ThreadPoolExecutor`.
- STEP spans created by reflectively wrapping each handler's
  `_query_FC` / `_query_prompting` (discovered via
  `bfcl_eval.constants.model_config.MODEL_CONFIG_MAPPING`).
- Per-call TOOL spans emitted by wrapping
  `bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils.execute_multi_turn_func_call`.
- Provider override mapping for OSS handlers (vLLM / SGLang).
- Multi-turn `bfcl.turn_idx` and ReAct `gen_ai.react.round` tracking via
  `contextvars`.

## Version 0.1.3.dev0 (2026-05-28)

There are no changelog entries for this release.
