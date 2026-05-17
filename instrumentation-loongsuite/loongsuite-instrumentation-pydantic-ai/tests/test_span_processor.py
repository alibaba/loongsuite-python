from opentelemetry.instrumentation.pydantic_ai.span_processor import (
    GEN_AI_FRAMEWORK,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_TOTAL_TOKENS,
    normalize_genai_attributes,
)


def test_normalize_genai_attributes_adds_framework_and_span_kind():
    attrs = normalize_genai_attributes(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.usage.input_tokens": 3,
            "gen_ai.usage.output_tokens": 5,
        }
    )

    assert attrs[GEN_AI_FRAMEWORK] == "pydantic-ai"
    assert attrs[GEN_AI_SPAN_KIND] == "LLM"
    assert attrs[GEN_AI_USAGE_TOTAL_TOKENS] == 8


def test_normalize_genai_attributes_preserves_existing_values():
    attrs = normalize_genai_attributes(
        {
            "gen_ai.operation.name": "execute_tool",
            GEN_AI_FRAMEWORK: "custom",
            GEN_AI_SPAN_KIND: "RETRIEVER",
            GEN_AI_USAGE_TOTAL_TOKENS: 10,
            "gen_ai.usage.input_tokens": 1,
            "gen_ai.usage.output_tokens": 2,
        }
    )

    assert attrs[GEN_AI_FRAMEWORK] == "custom"
    assert attrs[GEN_AI_SPAN_KIND] == "RETRIEVER"
    assert attrs[GEN_AI_USAGE_TOTAL_TOKENS] == 10
