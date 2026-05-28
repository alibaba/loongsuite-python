# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper classes for the BFCL v4 instrumentation.

Each wrapper follows the standard ``wrapt`` callable contract::

    def __call__(self, wrapped, instance, args, kwargs):
        ...

All wrappers rely on :func:`get_extended_telemetry_handler` (LoongSuite
``util-genai``) to create the actual spans, so that ENTRY / AGENT / STEP /
TOOL spans get the canonical ``gen_ai.span.kind`` and operation-name values
that the LoongSuite semantic-validator expects.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import logging
import os
import sys
import time
from contextvars import ContextVar
from typing import Any, Callable, Dict, Iterable, List, Optional

from opentelemetry.instrumentation.bfclv4.internal.attributes import (
    BFCL_NUM_THREADS,
    BFCL_QUERY_MODE,
    BFCL_RUN_IDS,
    BFCL_TEST_CASE_COUNT,
    BFCL_TEST_CATEGORY,
    BFCL_TEST_ENTRY_ID,
    BFCL_TOOL_DURATION_IS_ESTIMATED,
    BFCL_TOOL_INDEX,
    BFCL_TURN_IDX,
    FRAMEWORK_NAME,
    GEN_AI_FRAMEWORK,
    GEN_AI_PROVIDER_NAME,
)
from opentelemetry.instrumentation.bfclv4.internal.provider import (
    OSS_BACKEND_ENV,
    infer_provider,
)
from opentelemetry.instrumentation.bfclv4.internal.state import (
    bump_round,
    bump_turn,
    init_state,
    next_tool_index,
    reset_state,
)
from opentelemetry.instrumentation.bfclv4.internal.threading_propagation import (
    ContextPropagatingExecutor,
)
from opentelemetry.instrumentation.bfclv4.utils import (
    GenAIHookHelper,
    to_text_input,
    to_text_output,
    truncate_text,
)
from opentelemetry.util.genai.extended_handler import (
    get_extended_telemetry_handler,
)
from opentelemetry.util.genai.extended_types import (
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    GenericToolDefinition,
    InputMessage,
    OutputMessage,
    Text,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _flatten_tokens(value: Any) -> Optional[int]:
    """Sum a possibly nested ``int|float|list|list[list]`` BFCL token field."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, Iterable):
        total = 0
        any_seen = False
        for item in value:
            sub = _flatten_tokens(item)
            if sub is not None:
                total += sub
                any_seen = True
        if any_seen:
            return total
    return None


def _test_category_from_id(test_entry_id: Optional[str]) -> Optional[str]:
    if not test_entry_id or "_" not in test_entry_id:
        return None
    return test_entry_id.rsplit("_", 1)[0]


def _join_test_category(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        joined = ",".join(str(v) for v in value if v is not None)
        return joined or None
    return str(value)


BFCLV4_DEBUG_ENV = "BFCLV4_DEBUG"
GEN_AI_INPUT_MESSAGES_ATTR = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES_ATTR = "gen_ai.output.messages"
GEN_AI_SYSTEM_INSTRUCTIONS_ATTR = "gen_ai.system_instructions"
GEN_AI_TOOL_CALL_ARGUMENTS_ATTR = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT_ATTR = "gen_ai.tool.call.result"
GEN_AI_TOOL_CALL_ID_ATTR = "gen_ai.tool.call.id"
GEN_AI_TOOL_NAME_ATTR = "gen_ai.tool.name"
GEN_AI_TOOL_TYPE_ATTR = "gen_ai.tool.type"
GEN_AI_TOOL_DESCRIPTION_ATTR = "gen_ai.tool.description"
BFCL_SYNTHETIC_TOOL_CALL = "bfcl.tool.synthetic_from_model_response"
_TOOL_DESCRIPTION_MAP: ContextVar[dict[str, str]] = ContextVar(
    "bfclv4_tool_description_map", default={}
)


def _json_attr(value: Any) -> str:
    try:
        import json

        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return _safe_str(value)


def _message_dict(role: str, content: Any) -> dict:
    return {
        "role": role,
        "parts": [
            {"type": "text", "content": truncate_text(_safe_str(content))}
        ],
    }


def _system_instruction_dict(content: Any) -> dict:
    return {"type": "text", "content": truncate_text(_safe_str(content))}


class _BFCLCapturedError(RuntimeError):
    """Synthetic exception that surfaces BFCL-captured error strings on spans.

    BFCL's outer ``multi_threaded_inference`` swallows real exceptions and
    converts them into ``"Error during inference: ..."`` strings; the same
    happens for tool execution errors. We wrap those strings in this class so
    that ``span.record_exception`` produces a real exception event with the
    error message visible to span consumers.
    """


def _record_span_error(
    span: Any,
    error_text: str,
    *,
    exc_type: type = _BFCLCapturedError,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    if span is None:
        return
    try:
        if not span.is_recording():
            return
    except Exception:  # noqa: BLE001
        return
    try:
        from opentelemetry.trace import Status, StatusCode
    except Exception:  # noqa: BLE001
        return
    exc = exc_type(error_text)
    try:
        span.record_exception(exc, attributes=attributes or None)
    except Exception:  # noqa: BLE001
        logger.debug("bfclv4: record_exception failed", exc_info=True)
    try:
        span.set_status(Status(StatusCode.ERROR, error_text[:200]))
    except Exception:  # noqa: BLE001
        logger.debug("bfclv4: set_status ERROR failed", exc_info=True)


def _normalise_role(value: Any, default: str) -> str:
    if value in (None, "", [], {}):
        return default
    role = str(value)
    return role or default


def _normalise_message_dict(item: Any, *, default_role: str) -> Optional[dict]:
    """Convert a single BFCL message-like value to ``{role, parts:[{type,content}]}``.

    Returns ``None`` for empty values so callers can skip them.
    """
    if item in (None, "", [], {}):
        return None
    if isinstance(item, dict):
        role = _normalise_role(item.get("role"), default_role)
        content = item.get("content")
        if content in (None, "", [], {}):
            extras = {
                k: v
                for k, v in item.items()
                if k not in {"role", "name", "tool_call_id"}
            }
            content = extras if extras else None
        if content in (None, "", [], {}):
            return None
        text = truncate_text(_safe_str(content))
        return {"role": role, "parts": [{"type": "text", "content": text}]}
    text = truncate_text(_safe_str(item))
    return {"role": default_role, "parts": [{"type": "text", "content": text}]}


def _flatten_messages(value: Any, default_role: str = "user") -> List[dict]:
    """Flatten arbitrary BFCL question/answer structures into a list of message dicts.

    BFCL stores multi-turn questions as ``[[{...}, {...}], [{...}]]`` (list of
    turns, each turn a list of role/content dicts). Single-turn entries are
    ``[{...}]`` or even a bare dict/string. We flatten everything one level so
    each role/content pair becomes its own ``{role, parts:[{type,content}]}``
    message — avoiding the previous behaviour where the whole nested list was
    JSON-stringified into a single message's ``content`` field.
    """
    messages: List[dict] = []
    if value in (None, "", [], {}):
        return messages
    if isinstance(value, dict):
        msg = _normalise_message_dict(value, default_role=default_role)
        if msg is not None:
            messages.append(msg)
        return messages
    if isinstance(value, (list, tuple)):
        for item in value:
            messages.extend(_flatten_messages(item, default_role))
        return messages
    msg = _normalise_message_dict(value, default_role=default_role)
    if msg is not None:
        messages.append(msg)
    return messages


def _messages_to_input(messages: List[dict]) -> List[InputMessage]:
    result: List[InputMessage] = []
    for msg in messages:
        parts = [
            Text(content=p.get("content", "")) for p in msg.get("parts", [])
        ]
        if not parts:
            continue
        result.append(InputMessage(role=msg.get("role", "user"), parts=parts))
    return result


def _messages_to_output(
    messages: List[dict], finish_reason: str = "stop"
) -> List[OutputMessage]:
    result: List[OutputMessage] = []
    for msg in messages:
        parts = [
            Text(content=p.get("content", "")) for p in msg.get("parts", [])
        ]
        if not parts:
            continue
        result.append(
            OutputMessage(
                role=msg.get("role", "assistant"),
                parts=parts,
                finish_reason=finish_reason,
            )
        )
    return result


def _test_entry_to_messages(test_entry: Any):
    if not isinstance(test_entry, dict):
        return [], []

    inputs = []
    system_instructions = []
    for key in (
        "system",
        "system_prompt",
        "system_instruction",
        "system_instructions",
    ):
        value = test_entry.get(key)
        if value not in (None, "", [], {}):
            system_instructions.append(
                Text(content=truncate_text(_safe_str(value)))
            )

    _append_question_messages(
        test_entry.get("question"),
        inputs,
        system_instructions,
    )
    return inputs, system_instructions


def _append_question_messages(
    value: Any,
    inputs: list,
    system_instructions: list,
) -> None:
    if value in (None, "", [], {}):
        return

    if isinstance(value, dict):
        role = str(value.get("role") or "user")
        content = value.get("content")
        if content in (None, "", [], {}):
            content = {
                k: v
                for k, v in value.items()
                if k not in {"role", "name", "tool_call_id"}
            }
        if content in (None, "", [], {}):
            return
        text = truncate_text(_safe_str(content))
        if role == "system":
            system_instructions.append(Text(content=text))
        else:
            inputs.extend(to_text_input(role, text))
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            _append_question_messages(item, inputs, system_instructions)
        return

    inputs.extend(to_text_input("user", truncate_text(_safe_str(value))))


def _test_entry_to_tool_definitions(test_entry: Any) -> list:
    if not isinstance(test_entry, dict):
        return []

    definitions = []
    for key in ("function", "functions", "tools", "tool_definitions"):
        definitions.extend(_tool_value_to_definitions(test_entry.get(key)))

    missed_function = test_entry.get("missed_function")
    if isinstance(missed_function, dict):
        for value in missed_function.values():
            definitions.extend(_tool_value_to_definitions(value))
    else:
        definitions.extend(_tool_value_to_definitions(missed_function))

    return _dedupe_tool_definitions(definitions)


def _tool_value_to_definitions(value: Any) -> list:
    if value in (None, "", [], {}):
        return []

    if isinstance(value, str):
        try:
            import json

            value = json.loads(value)
        except Exception:  # noqa: BLE001
            return []

    if isinstance(value, (list, tuple)):
        definitions = []
        for item in value:
            definitions.extend(_tool_value_to_definitions(item))
        return definitions

    if not isinstance(value, dict):
        return []

    nested_function = value.get("function")
    if isinstance(nested_function, dict):
        nested = dict(nested_function)
        nested.setdefault("type", value.get("type", "function"))
        return _tool_value_to_definitions(nested)

    name = (
        value.get("name")
        or value.get("function_name")
        or value.get("tool_name")
    )
    if not name:
        return []

    tool_type = value.get("type")
    description = value.get("description")
    parameters = value.get("parameters")
    if tool_type not in (None, "", "function") and parameters is None:
        return [GenericToolDefinition(name=str(name), type=str(tool_type))]

    return [
        FunctionToolDefinition(
            name=str(name),
            description=_safe_str(description)
            if description is not None
            else None,
            parameters=parameters,
        )
    ]


def _dedupe_tool_definitions(definitions: list) -> list:
    deduped = []
    seen = set()
    for definition in definitions:
        key = _json_attr(getattr(definition, "__dict__", repr(definition)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(definition)
    return deduped


def _tool_description_map(test_entry: Any) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for definition in _test_entry_to_tool_definitions(test_entry):
        name = getattr(definition, "name", None)
        description = getattr(definition, "description", None)
        if name and description:
            descriptions[str(name)] = _safe_str(description)

    # Multi-turn BFCL cases often leave ``function`` empty and expose tools via
    # involved_classes. Pull method docstrings from BFCL's executable classes so
    # TOOL spans still carry gen_ai.tool.description.
    if isinstance(test_entry, dict):
        involved_classes = test_entry.get("involved_classes") or []
        try:
            from bfcl_eval.constants.executable_backend_config import (  # noqa: PLC0415
                CLASS_FILE_PATH_MAPPING,
            )
        except Exception:  # noqa: BLE001
            CLASS_FILE_PATH_MAPPING = {}
        for class_name in (
            involved_classes
            if isinstance(involved_classes, (list, tuple))
            else []
        ):
            module_name = CLASS_FILE_PATH_MAPPING.get(class_name)
            if not module_name:
                continue
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
            except Exception:  # noqa: BLE001
                continue
            for method_name, method in inspect.getmembers(
                cls, predicate=inspect.isfunction
            ):
                if method_name.startswith("_") or method_name in descriptions:
                    continue
                doc = inspect.getdoc(method)
                if doc:
                    descriptions[method_name] = truncate_text(doc, 1024)
    return descriptions


def _lookup_tool_description(tool_name: Optional[str]) -> Optional[str]:
    if not tool_name:
        return None
    description = _TOOL_DESCRIPTION_MAP.get().get(str(tool_name))
    if description:
        return description
    try:
        from bfcl_eval.constants.executable_backend_config import (  # noqa: PLC0415
            CLASS_FILE_PATH_MAPPING,
        )
    except Exception:  # noqa: BLE001
        CLASS_FILE_PATH_MAPPING = {}
    for module_name in CLASS_FILE_PATH_MAPPING.values():
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            method = getattr(cls, str(tool_name), None)
            if method is None:
                continue
            doc = inspect.getdoc(method)
            if doc:
                return truncate_text(doc, 1024)
    return None


def _normalise_tool_arguments(arguments: Any) -> Any:
    return {} if arguments is None else arguments


def _extract_questions_from_cases(cases: Any) -> list:
    if not isinstance(cases, (list, tuple)):
        return []
    messages: list = []
    for case in cases[:10]:
        if isinstance(case, dict) and case.get("question") is not None:
            messages.extend(_flatten_messages(case.get("question"), "user"))
    return messages


def _extract_tool_defs_from_cases(cases: Any) -> list:
    if not isinstance(cases, (list, tuple)):
        return []
    instructions = []
    for case in cases[:10]:
        if isinstance(case, dict) and case.get("function") is not None:
            instructions.append(_system_instruction_dict(case.get("function")))
    return instructions


def _set_json_span_attr(span: Any, key: str, value: Any) -> None:
    if not value or span is None:
        return
    try:
        if span.is_recording():
            span.set_attribute(key, _json_attr(value))
    except Exception:  # noqa: BLE001
        logger.debug("bfclv4: failed to set json attr %s", key, exc_info=True)


def _span_attr_value(value: Any) -> str:
    return value if isinstance(value, str) else _json_attr(value)


def _set_tool_call_span_attrs(
    span: Any,
    *,
    arguments: Any = None,
    result: Any = None,
    description: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    tool_type: Optional[str] = "function",
) -> None:
    if span is None:
        return
    try:
        if not span.is_recording():
            return
        if tool_call_id:
            span.set_attribute(GEN_AI_TOOL_CALL_ID_ATTR, tool_call_id)
        if tool_name:
            span.set_attribute(GEN_AI_TOOL_NAME_ATTR, tool_name)
        if tool_type:
            span.set_attribute(GEN_AI_TOOL_TYPE_ATTR, tool_type)
        if arguments is not None:
            span.set_attribute(
                GEN_AI_TOOL_CALL_ARGUMENTS_ATTR,
                _span_attr_value(arguments),
            )
        if result is not None:
            span.set_attribute(
                GEN_AI_TOOL_CALL_RESULT_ATTR,
                _span_attr_value(result),
            )
        if description:
            span.set_attribute(GEN_AI_TOOL_DESCRIPTION_ATTR, description)
        print(
            "[bfclv4-tool-attrs] "
            f"name={tool_name} id={tool_call_id} "
            f"has_arguments={arguments is not None} "
            f"has_result={result is not None} "
            f"has_description={bool(description)}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:  # noqa: BLE001
        logger.debug("bfclv4: failed to set TOOL call attrs", exc_info=True)


def _parse_python_call_arguments(func_call: Any) -> Any:
    if not isinstance(func_call, str) or "(" not in func_call:
        return _extract_tool_arguments(func_call)
    try:
        expr = ast.parse(func_call, mode="eval").body
    except SyntaxError:
        return _extract_tool_arguments(func_call)
    if not isinstance(expr, ast.Call):
        return _extract_tool_arguments(func_call)

    parsed: dict[str, Any] = {}
    for index, arg in enumerate(expr.args):
        parsed[f"arg_{index}"] = _literal_or_source(arg, func_call)
    for keyword in expr.keywords:
        if keyword.arg is None:
            parsed["kwargs"] = _literal_or_source(keyword.value, func_call)
        else:
            parsed[keyword.arg] = _literal_or_source(keyword.value, func_call)
    return parsed or None


def _literal_or_source(node: ast.AST, source: str) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        segment = ast.get_source_segment(source, node)
        return segment if segment is not None else _safe_str(node)


def _iter_model_tool_calls(result_payload: Any):
    """Yield (tool_name, arguments) pairs from BFCL single-turn decoded output."""
    if not isinstance(result_payload, list):
        return
    for item in result_payload:
        if isinstance(item, dict):
            for name, arguments in item.items():
                yield str(name), arguments
        elif isinstance(item, str):
            yield _extract_tool_name(item), _parse_python_call_arguments(item)


def _emit_synthetic_tool_spans(
    result_payload: Any,
    *,
    test_entry_id: Optional[Any],
    model_name: Optional[Any],
) -> int:
    """Emit TOOL spans for BFCL cases that generate calls but do not execute them."""
    calls = list(_iter_model_tool_calls(result_payload) or [])
    if not calls:
        return 0
    handler_obj = get_extended_telemetry_handler()
    emitted = 0
    for index, (tool_name, arguments) in enumerate(calls):
        description = _lookup_tool_description(tool_name)
        tool_inv = ExecuteToolInvocation(
            tool_name=tool_name or "unknown",
            tool_call_id=_synth_tool_call_id(test_entry_id, model_name, index),
            tool_type="function",
            tool_description=description,
            tool_call_arguments=_normalise_tool_arguments(arguments),
            tool_call_result=None,
        )
        try:
            with handler_obj.execute_tool(tool_inv) as inv:
                span = inv.span
                if span is not None and span.is_recording():
                    span.set_attribute(GEN_AI_FRAMEWORK, FRAMEWORK_NAME)
                    span.set_attribute(BFCL_TOOL_INDEX, index)
                    span.set_attribute(BFCL_SYNTHETIC_TOOL_CALL, True)
                    if test_entry_id is not None:
                        span.set_attribute(
                            BFCL_TEST_ENTRY_ID, str(test_entry_id)
                        )
                    _set_tool_call_span_attrs(
                        span,
                        arguments=_normalise_tool_arguments(arguments),
                        description=description,
                        tool_name=tool_name,
                        tool_call_id=_synth_tool_call_id(
                            test_entry_id, model_name, index
                        ),
                        tool_type="function",
                    )
            emitted += 1
        except Exception:  # noqa: BLE001
            logger.debug(
                "bfclv4 synthetic TOOL span emission failed", exc_info=True
            )
    return emitted


# ---------------------------------------------------------------------------
# ENTRY wrapper


class GenerateResultsWrapper:
    """Wraps ``bfcl_eval._llm_response_generation.generate_results``.

    Responsibilities:

    * Open the ENTRY span (``enter_ai_application_system``).
    * Temporarily swap the ``ThreadPoolExecutor`` reference inside the BFCL
      generation module to a context-propagating subclass so that AGENT spans
      created in worker threads inherit the ENTRY span as parent.
    * Publish ``args.backend`` to ``BFCL_BACKEND`` so that
      :func:`infer_provider` can attribute OSS spans to vllm / sglang.
    """

    def __init__(self, helper: GenAIHookHelper) -> None:
        self._helper = helper

    def __call__(self, wrapped: Callable, instance: Any, args, kwargs):  # noqa: D401
        # ``generate_results(args, model_name, test_cases_total)``
        cli_args = args[0] if len(args) >= 1 else kwargs.get("args")
        model_name = args[1] if len(args) >= 2 else kwargs.get("model_name")
        test_cases_total = (
            args[2] if len(args) >= 3 else kwargs.get("test_cases_total")
        )

        try:
            from bfcl_eval import (  # noqa: PLC0415
                _llm_response_generation as _bfcl_gen,
            )
        except ImportError:
            return wrapped(*args, **kwargs)

        original_executor = getattr(_bfcl_gen, "ThreadPoolExecutor", None)
        if original_executor is not None:
            _bfcl_gen.ThreadPoolExecutor = ContextPropagatingExecutor

        backend_value = (
            _safe_get(cli_args, "backend", None)
            if cli_args is not None
            else None
        )
        previous_backend_env = os.environ.get(OSS_BACKEND_ENV)
        if backend_value:
            os.environ[OSS_BACKEND_ENV] = str(backend_value)

        session_id_default = None
        if model_name is not None:
            try:
                session_id_default = f"{model_name}@{int(time.time())}"
            except Exception:  # noqa: BLE001
                session_id_default = None
        session_id = os.environ.get("BFCL_SESSION_ID") or session_id_default

        entry_inv = EntryInvocation(session_id=session_id)
        entry_input_messages = _extract_questions_from_cases(test_cases_total)
        entry_system_instructions = _extract_tool_defs_from_cases(
            test_cases_total
        )
        entry_inv.input_messages = _messages_to_input(entry_input_messages)
        handler = get_extended_telemetry_handler()

        attributes = {GEN_AI_FRAMEWORK: FRAMEWORK_NAME}
        category_value = _join_test_category(
            _safe_get(cli_args, "test_category", None)
        )
        if category_value:
            attributes[BFCL_TEST_CATEGORY] = category_value
        num_threads = _safe_get(cli_args, "num_threads", None)
        if num_threads is not None:
            try:
                attributes[BFCL_NUM_THREADS] = int(num_threads)
            except (TypeError, ValueError):
                pass
        if isinstance(test_cases_total, (list, tuple)):
            attributes[BFCL_TEST_CASE_COUNT] = len(test_cases_total)
        attributes[BFCL_RUN_IDS] = bool(_safe_get(cli_args, "run_ids", False))

        try:
            with handler.entry(entry_inv) as inv:
                if inv.span is not None and inv.span.is_recording():
                    for key, value in attributes.items():
                        try:
                            inv.span.set_attribute(key, value)
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "bfclv4 ENTRY set_attribute(%s) failed",
                                key,
                                exc_info=True,
                            )
                    _set_json_span_attr(
                        inv.span,
                        GEN_AI_INPUT_MESSAGES_ATTR,
                        entry_input_messages,
                    )
                    _set_json_span_attr(
                        inv.span,
                        GEN_AI_SYSTEM_INSTRUCTIONS_ATTR,
                        entry_system_instructions,
                    )
                try:
                    result = wrapped(*args, **kwargs)
                except Exception as exc:
                    if inv.span is not None and inv.span.is_recording():
                        try:
                            inv.span.record_exception(exc)
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "bfclv4 ENTRY: record_exception failed",
                                exc_info=True,
                            )
                    raise
                if inv.span is not None and inv.span.is_recording():
                    _set_json_span_attr(
                        inv.span,
                        GEN_AI_OUTPUT_MESSAGES_ATTR,
                        [
                            _message_dict(
                                "assistant",
                                {
                                    "model": model_name,
                                    "status": "generate_results_completed",
                                },
                            )
                        ],
                    )
                return result
        finally:
            if original_executor is not None:
                try:
                    _bfcl_gen.ThreadPoolExecutor = original_executor
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "bfclv4 ENTRY: failed to restore ThreadPoolExecutor",
                        exc_info=True,
                    )
            if backend_value:
                if previous_backend_env is None:
                    os.environ.pop(OSS_BACKEND_ENV, None)
                else:
                    os.environ[OSS_BACKEND_ENV] = previous_backend_env


# ---------------------------------------------------------------------------
# AGENT wrapper


_BFCL_INFERENCE_ERROR_PREFIX = "Error during inference:"


class BaseHandlerInferenceWrapper:
    """Wraps ``BaseHandler.inference``.

    Creates the AGENT span (kind=AGENT, op=invoke_agent) and initialises the
    per-thread ReAct state used by the STEP wrapper.

    BFCL's outer ``multi_threaded_inference`` catches every exception and
    converts it into a ``"Error during inference: ..."`` string; we mirror
    that behaviour by setting the AGENT span status to ERROR when the
    returned ``result`` looks like an error string, instead of relying on
    a re-raised exception.
    """

    def __init__(self, helper: GenAIHookHelper) -> None:
        self._helper = helper

    def __call__(self, wrapped: Callable, instance: Any, args, kwargs):  # noqa: D401
        # ``inference(self, test_entry, include_input_log, exclude_state_log)``
        test_entry = args[0] if args else kwargs.get("test_entry")
        if not isinstance(test_entry, dict):
            return wrapped(*args, **kwargs)

        provider, extra_attrs = infer_provider(instance)
        request_model = getattr(instance, "model_name", None)
        test_entry_id = test_entry.get("id")
        category = _test_category_from_id(test_entry_id)
        involved_classes = test_entry.get("involved_classes") or []
        agent_description = (
            ", ".join(str(c) for c in involved_classes)
            if isinstance(involved_classes, (list, tuple))
            else None
        )

        invocation = InvokeAgentInvocation(
            provider=provider or "unknown",
            request_model=request_model,
            agent_id=test_entry_id,
            agent_name=category or "bfcl_agent",
            agent_description=agent_description or None,
            conversation_id=test_entry_id,
        )

        token = init_state()
        tool_description_token = _TOOL_DESCRIPTION_MAP.set(
            _tool_description_map(test_entry)
        )
        handler = get_extended_telemetry_handler()
        try:
            with handler.invoke_agent(invocation) as inv:
                if inv.span is not None and inv.span.is_recording():
                    inv.span.set_attribute(GEN_AI_FRAMEWORK, FRAMEWORK_NAME)
                    if provider:
                        inv.span.set_attribute(GEN_AI_PROVIDER_NAME, provider)
                    if test_entry_id is not None:
                        inv.span.set_attribute(
                            BFCL_TEST_ENTRY_ID, test_entry_id
                        )
                    if category is not None:
                        inv.span.set_attribute(BFCL_TEST_CATEGORY, category)
                    for key, value in extra_attrs.items():
                        if value is not None:
                            inv.span.set_attribute(key, value)

                # Capture inputs for the AGENT. Also write span attributes directly
                # because util-genai gates message attributes behind experimental
                # content-capture mode, which makes K8s semantic validation opaque.
                question = test_entry.get("question")
                functions = test_entry.get("function")
                input_messages_dicts = _flatten_messages(question, "user")
                if input_messages_dicts:
                    inv.input_messages = _messages_to_input(
                        input_messages_dicts
                    )
                if functions is not None:
                    system_inputs = to_text_input(
                        "system", truncate_text(_safe_str(functions))
                    )
                    inv.system_instruction = (
                        system_inputs[0].parts if system_inputs else []
                    )
                if inv.span is not None and inv.span.is_recording():
                    if input_messages_dicts:
                        _set_json_span_attr(
                            inv.span,
                            GEN_AI_INPUT_MESSAGES_ATTR,
                            input_messages_dicts,
                        )
                    if functions is not None:
                        _set_json_span_attr(
                            inv.span,
                            GEN_AI_SYSTEM_INSTRUCTIONS_ATTR,
                            [_system_instruction_dict(functions)],
                        )
                # Run the original inference call.
                try:
                    result = wrapped(*args, **kwargs)
                except Exception as exc:
                    # The CM will mark the span as failed; record the
                    # exception explicitly so the traceback/message is visible
                    # on the span (util-genai's fail path only sets status).
                    if inv.span is not None and inv.span.is_recording():
                        try:
                            inv.span.record_exception(exc)
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "bfclv4 AGENT: record_exception failed",
                                exc_info=True,
                            )
                    raise

                # Detect BFCL's own captured error path (no exception raised
                # but the returned result is the error string).
                result_payload = (
                    result[0] if isinstance(result, tuple) and result else None
                )
                metadata_payload = (
                    result[1]
                    if isinstance(result, tuple) and len(result) >= 2
                    else None
                )

                if isinstance(
                    result_payload, str
                ) and result_payload.startswith(_BFCL_INFERENCE_ERROR_PREFIX):
                    _record_span_error(
                        inv.span,
                        result_payload,
                        attributes={"bfcl.error.captured": True},
                    )

                if isinstance(metadata_payload, dict):
                    input_tokens = _flatten_tokens(
                        metadata_payload.get("input_token_count")
                    )
                    output_tokens = _flatten_tokens(
                        metadata_payload.get("output_token_count")
                    )
                    if input_tokens is not None:
                        inv.input_tokens = input_tokens
                    if output_tokens is not None:
                        inv.output_tokens = output_tokens

                if result_payload is not None:
                    output_messages_dicts = _flatten_messages(
                        result_payload, "assistant"
                    )
                    if not output_messages_dicts:
                        output_messages_dicts = [
                            _message_dict("assistant", result_payload)
                        ]
                    inv.output_messages = _messages_to_output(
                        output_messages_dicts
                    )
                    if inv.span is not None and inv.span.is_recording():
                        _set_json_span_attr(
                            inv.span,
                            GEN_AI_OUTPUT_MESSAGES_ATTR,
                            output_messages_dicts,
                        )

                _emit_synthetic_tool_spans(
                    result_payload,
                    test_entry_id=test_entry_id,
                    model_name=request_model,
                )

                return result
        finally:
            try:
                _TOOL_DESCRIPTION_MAP.reset(tool_description_token)
            except (LookupError, ValueError):
                pass
            reset_state(token)


def _safe_str(value: Any) -> str:
    try:
        if isinstance(value, str):
            return value
        import json

        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        try:
            return str(value)
        except Exception:  # noqa: BLE001
            return "<unserialisable>"


def _result_to_output_messages(result: Any):
    payload = result[0] if isinstance(result, tuple) and result else result
    if payload in (None, "", [], {}):
        return []

    if isinstance(payload, (list, tuple)):
        messages = []
        for item in payload:
            messages.extend(_result_to_output_messages(item))
        return messages

    content = _extract_result_content(payload)
    if content in (None, "", [], {}):
        return []
    return to_text_output("assistant", truncate_text(_safe_str(content)))


def _extract_result_content(result: Any) -> Any:
    if not isinstance(result, dict):
        return result

    for key in (
        "final_answer",
        "answer",
        "output",
        "result",
        "model_response",
        "model_responses",
        "inference_output",
    ):
        value = result.get(key)
        if value not in (None, "", [], {}):
            return value

    inference_log = result.get("inference_log")
    if isinstance(inference_log, dict):
        for key in sorted(
            (k for k in inference_log if k.startswith("step_")),
            key=_step_log_sort_key,
            reverse=True,
        ):
            step_data = inference_log.get(key)
            if not isinstance(step_data, dict):
                continue
            output = step_data.get("inference_output")
            if output not in (None, "", [], {}):
                return output
            answer = step_data.get("inference_answer")
            if answer not in (None, "", [], {}):
                return answer

    return result


def _step_log_sort_key(key: str) -> int:
    try:
        return int(key[len("step_") :])
    except (TypeError, ValueError):
        return -1


# ---------------------------------------------------------------------------
# STEP wrapper


class QueryWrapper:
    """Wraps ``<Handler>._query_FC`` / ``_query_prompting``.

    Creates a ReAct STEP span, attaches token usage by re-calling the
    handler's matching ``_parse_query_response_*`` (which is documented as
    side-effect-free).
    """

    def __init__(self, helper: GenAIHookHelper, mode: str) -> None:
        self._helper = helper
        self._mode = mode  # "FC" or "prompting"

    def __call__(self, wrapped: Callable, instance: Any, args, kwargs):  # noqa: D401
        round_idx = bump_round()
        provider, extra_attrs = infer_provider(instance)

        invocation = ReactStepInvocation(round=round_idx)
        handler_obj = get_extended_telemetry_handler()
        with handler_obj.react_step(invocation) as step_inv:
            span = step_inv.span
            if span is not None and span.is_recording():
                span.set_attribute(GEN_AI_FRAMEWORK, FRAMEWORK_NAME)
                span.set_attribute(BFCL_QUERY_MODE, self._mode)
                if provider:
                    span.set_attribute(GEN_AI_PROVIDER_NAME, provider)
                model_name = getattr(instance, "model_name", None)
                if model_name:
                    span.set_attribute("gen_ai.request.model", str(model_name))
                from opentelemetry.instrumentation.bfclv4.internal.state import (
                    get_state,
                )

                state = get_state()
                if state is not None:
                    span.set_attribute(BFCL_TURN_IDX, state.get("turn_idx", 0))
                for key, value in extra_attrs.items():
                    if value is not None:
                        span.set_attribute(key, value)

            try:
                api_response, query_latency = wrapped(*args, **kwargs)
            except Exception:
                # Let the context-manager mark the span as failed; the BFCL
                # outer try/except will turn this into an "Error during
                # inference: ..." result string at the AGENT layer.
                raise

            # When the underlying handler returns a streaming wrapper
            # (e.g. ``ChatStreamWrapper`` from openai-v2), the LLM span and
            # its OTel context attach are kept alive until the stream is
            # consumed by BFCL's ``_parse_query_response_*`` *outside* of
            # this STEP context manager. That breaks the LIFO ordering of
            # context attach/detach, leaving the LLM span as the "current"
            # span after the STEP CM exits, which causes the next STEP and
            # any TOOL spans to be parented to the previous STEP rather
            # than to the AGENT.
            #
            # To preserve LIFO ordering, force-consume the stream here
            # (inside the STEP context) and replace it with a plain
            # iterator over the cached chunks. This makes ``stop_llm``
            # (which detaches the LLM context) run *before* STEP detaches.
            if (
                api_response is not None
                and hasattr(api_response, "__next__")
                and not isinstance(api_response, (str, bytes))
            ):
                try:
                    chunks = list(api_response)
                    api_response = iter(chunks)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "bfclv4 STEP: failed to materialise streaming "
                        "response; LLM/STEP nesting may be incorrect",
                        exc_info=True,
                    )

            # Post-call attribute enrichment - use try/except so that any
            # vendor-side parsing surprise never breaks BFCL itself.
            #
            # IMPORTANT: We must NOT re-call ``_parse_query_response_*`` here,
            # because for streaming providers (e.g. Qwen DashScope) the
            # ``api_response`` is a single-pass generator that the parser
            # consumes; calling it twice leaves BFCL's own subsequent call to
            # the parser with an exhausted iterator, which crashes inference
            # with ``UnboundLocalError: chunk``. Token usage will instead be
            # recovered later from the AGENT-level metadata payload.
            try:
                if span is not None and span.is_recording():
                    if isinstance(query_latency, (int, float)):
                        try:
                            span.set_attribute(
                                "gen_ai.response.time_to_first_token",
                                int(float(query_latency) * 1e9),
                            )
                        except Exception:  # noqa: BLE001
                            pass
            except Exception:  # noqa: BLE001
                logger.debug(
                    "bfclv4 STEP: post-call enrichment failed", exc_info=True
                )

            return api_response, query_latency


def _infer_finish_reason(model_responses: Any) -> str:
    """Best-effort heuristic for ``gen_ai.react.finish_reason``."""
    if model_responses is None:
        return "unknown"
    if isinstance(model_responses, list):
        if len(model_responses) == 0:
            return "empty_response"
        if len(model_responses) == 1 and not model_responses[0]:
            return "empty_response"
        return "tool_calls"
    if isinstance(model_responses, str):
        # Prompting models often return decoded strings even when there are
        # no tool calls - treat as "stop" so downstream callers know there is
        # no further work to do.
        return "stop"
    return "continue"


# ---------------------------------------------------------------------------
# turn_idx maintenance wrappers (no spans)


class TurnBumpWrapper:
    """Wraps ``<Handler>.add_first_turn_message_*`` and
    ``<Handler>._add_next_turn_user_message_*`` to keep ``bfcl.turn_idx`` in
    sync.  No spans are created here.
    """

    def __init__(self, *, reset: bool) -> None:
        self._reset = reset

    def __call__(self, wrapped: Callable, instance: Any, args, kwargs):  # noqa: D401
        try:
            if self._reset:
                # ``add_first_turn_message_*`` runs once at the very start of
                # multi-turn / single-turn inference.  We only want to reset
                # to ``turn_idx=0`` here.
                from opentelemetry.instrumentation.bfclv4.internal.state import (
                    get_state,
                )

                state = get_state()
                if state is not None:
                    state["turn_idx"] = 0
                    state["fc_round"] = 0
            else:
                bump_turn()
        except Exception:  # noqa: BLE001
            logger.debug("bfclv4: turn_idx maintenance failed", exc_info=True)
        return wrapped(*args, **kwargs)


# ---------------------------------------------------------------------------
# TOOL wrapper


class ExecuteFuncCallWrapper:
    """Wraps
    ``bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils.execute_multi_turn_func_call``.

    BFCL evaluates a list of function-call strings in a single Python call;
    we surface each one as its own TOOL span by post-processing the wrapped
    result.  Per-call latency is approximated by averaging the total elapsed
    time across the batch (``bfcl.tool.duration_is_estimated=true``).
    """

    def __init__(self, helper: GenAIHookHelper) -> None:
        self._helper = helper

    def __call__(self, wrapped: Callable, instance: Any, args, kwargs):  # noqa: D401
        # ``execute_multi_turn_func_call(func_call_list, initial_config,
        #                                involved_classes, model_name,
        #                                test_entry_id, long_context=False,
        #                                is_evaL_run=False)``
        func_call_list = args[0] if args else kwargs.get("func_call_list", [])
        model_name = args[3] if len(args) >= 4 else kwargs.get("model_name")
        test_entry_id = (
            args[4] if len(args) >= 5 else kwargs.get("test_entry_id")
        )

        if not isinstance(func_call_list, list) or not func_call_list:
            return wrapped(*args, **kwargs)

        t0 = time.perf_counter()
        try:
            result = wrapped(*args, **kwargs)
        finally:
            elapsed = max(time.perf_counter() - t0, 0.0)

        execution_results: List[str] = []
        if isinstance(result, tuple) and result:
            payload = result[0]
            if isinstance(payload, list):
                execution_results = list(payload)

        per_call_seconds = (
            elapsed / len(func_call_list) if func_call_list else 0.0
        )

        handler_obj = get_extended_telemetry_handler()
        for index, func_call in enumerate(func_call_list):
            tool_name = _extract_tool_name(func_call)
            arguments = _parse_python_call_arguments(func_call)
            description = _lookup_tool_description(tool_name)
            execution_result = (
                execution_results[index]
                if index < len(execution_results)
                else None
            )

            tool_inv = ExecuteToolInvocation(
                tool_name=tool_name,
                tool_call_id=_synth_tool_call_id(
                    test_entry_id, model_name, index
                ),
                tool_type="function",
                tool_description=description,
                tool_call_arguments=_normalise_tool_arguments(arguments),
                tool_call_result=execution_result,
            )

            try:
                with handler_obj.execute_tool(tool_inv) as inv:
                    span = inv.span
                    if span is not None and span.is_recording():
                        span.set_attribute(GEN_AI_FRAMEWORK, FRAMEWORK_NAME)
                        span.set_attribute(BFCL_TOOL_INDEX, index)
                        span.set_attribute(
                            BFCL_TOOL_DURATION_IS_ESTIMATED, True
                        )
                        if test_entry_id is not None:
                            span.set_attribute(
                                BFCL_TEST_ENTRY_ID, str(test_entry_id)
                            )
                        _set_tool_call_span_attrs(
                            span,
                            arguments=_normalise_tool_arguments(arguments),
                            result=execution_result,
                            description=description,
                            tool_name=tool_name,
                            tool_call_id=_synth_tool_call_id(
                                test_entry_id, model_name, index
                            ),
                            tool_type="function",
                        )
                        if isinstance(
                            execution_result, str
                        ) and execution_result.startswith(
                            "Error during execution:"
                        ):
                            _record_span_error(
                                span,
                                execution_result,
                                attributes={
                                    "bfcl.tool.error.captured": True,
                                    BFCL_TOOL_INDEX: index,
                                },
                            )
                        # Approximate latency by sleeping the budgeted slice
                        # would distort BFCL execution; we instead rely on
                        # span start/end (currently both wall-clock-now).
                        # The ``bfcl.tool.duration_is_estimated`` attribute
                        # signals the limitation to consumers.
                        _ = per_call_seconds  # unused but documented
                # Bump a per-AGENT counter for downstream debugging.
                next_tool_index()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "bfclv4 TOOL: span emission failed for %s",
                    tool_name,
                    exc_info=True,
                )

        return result


def _extract_tool_name(func_call: Any) -> str:
    if not isinstance(func_call, str) or "(" not in func_call:
        return "unknown"
    head = func_call.split("(", 1)[0]
    # ``head`` may be ``module.method`` or ``instance.method`` - keep the
    # last segment which is the actual callable.
    return head.split(".")[-1] or "unknown"


def _extract_tool_arguments(func_call: Any) -> Optional[str]:
    if not isinstance(func_call, str):
        return None
    if "(" not in func_call or not func_call.endswith(")"):
        return func_call
    args_part = func_call[func_call.index("(") + 1 : -1]
    return args_part if args_part else None


def _synth_tool_call_id(
    test_entry_id: Optional[Any], model_name: Optional[Any], index: int
) -> str:
    parts = [
        str(test_entry_id) if test_entry_id is not None else "no_id",
        str(model_name) if model_name is not None else "no_model",
        str(index),
    ]
    return "-".join(parts)
