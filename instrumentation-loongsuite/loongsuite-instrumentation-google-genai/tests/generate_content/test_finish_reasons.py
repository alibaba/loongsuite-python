# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0


from google.genai import types as genai_types

from .base import TestCase


class FinishReasonsTestCase(TestCase):
    def generate_and_get_span_finish_reasons(self):
        self.client.models.generate_content(
            model="gemini-2.5-flash-001", contents="Some prompt"
        )
        span = self.otel.get_span_named(
            "generate_content gemini-2.5-flash-001"
        )
        assert span is not None
        if "gen_ai.response.finish_reasons" not in span.attributes:
            return []
        return list(span.attributes["gen_ai.response.finish_reasons"])

    def test_single_candidate_with_valid_reason(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.STOP
            )
        )
        self.assertEqual(self.generate_and_get_span_finish_reasons(), ["stop"])

    def test_single_candidate_with_safety_reason(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.SAFETY
            )
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(), ["safety"]
        )

    def test_single_candidate_with_max_tokens_reason(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.MAX_TOKENS
            )
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(), ["max_tokens"]
        )

    def test_single_candidate_with_no_reason(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(finish_reason=None)
        )
        self.assertEqual(self.generate_and_get_span_finish_reasons(), [])

    def test_single_candidate_with_unspecified_reason(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.FINISH_REASON_UNSPECIFIED
            )
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(),
            ["finish_reason_unspecified"],
        )

    def test_multiple_candidates_with_valid_reasons(self):
        self.configure_valid_response(
            candidates=[
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.MAX_TOKENS
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
            ]
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(), ["max_tokens", "stop"]
        )

    def test_doesnt_sort_finish_reasons(self):
        self.configure_valid_response(
            candidates=[
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.MAX_TOKENS
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.SAFETY
                ),
            ]
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(),
            ["stop", "max_tokens", "safety"],
        )

    def test_doesnt_deduplicate_finish_reasons(self):
        self.configure_valid_response(
            candidates=[
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.MAX_TOKENS
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.SAFETY
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
            ]
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(),
            [
                "stop",
                "max_tokens",
                "stop",
                "stop",
                "safety",
                "stop",
                "stop",
                "stop",
            ],
        )
