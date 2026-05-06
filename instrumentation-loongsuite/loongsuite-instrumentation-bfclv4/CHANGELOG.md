# Changelog

All notable changes to the LoongSuite BFCL v4 instrumentation are documented
in this file.

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
