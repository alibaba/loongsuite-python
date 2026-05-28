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

"""Tests for ``internal/provider.py`` -- provider detection logic."""

from __future__ import annotations


class TestInferProvider:
    def test_openai_completions(self):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            model_style = ModelStyle.OPENAI_COMPLETIONS

        name, extras = infer_provider(_Handler())
        assert name == "openai"
        assert extras == {}

    def test_openai_responses(self):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            model_style = ModelStyle.OPENAI_RESPONSES

        name, _ = infer_provider(_Handler())
        assert name == "openai"

    def test_anthropic(self):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            model_style = ModelStyle.ANTHROPIC

        name, _ = infer_provider(_Handler())
        assert name == "anthropic"

    def test_google(self):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            model_style = ModelStyle.GOOGLE

        name, _ = infer_provider(_Handler())
        assert name == "gcp.gemini"

    def test_mistral(self):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            model_style = ModelStyle.MISTRAL

        name, _ = infer_provider(_Handler())
        assert name == "mistral_ai"

    def test_cohere(self):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            model_style = ModelStyle.COHERE

        name, _ = infer_provider(_Handler())
        assert name == "cohere"

    def test_amazon(self):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            model_style = ModelStyle.AMAZON

        name, _ = infer_provider(_Handler())
        assert name == "aws.bedrock"

    def test_firework_ai(self):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            model_style = ModelStyle.FIREWORK_AI

        name, _ = infer_provider(_Handler())
        assert name == "fireworks_ai"

    def test_writer(self):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            model_style = ModelStyle.WRITER

        name, _ = infer_provider(_Handler())
        assert name == "writer"

    def test_novita_ai(self):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            model_style = ModelStyle.NOVITA_AI

        name, _ = infer_provider(_Handler())
        assert name == "novita"

    def test_nexus(self):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            model_style = ModelStyle.NEXUS

        name, _ = infer_provider(_Handler())
        assert name == "nexusflow"

    def test_gorilla(self):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            model_style = ModelStyle.GORILLA

        name, _ = infer_provider(_Handler())
        assert name == "gorilla"

    def test_oss_vllm(self, monkeypatch):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        monkeypatch.setenv("BFCL_BACKEND", "vllm")

        class _Handler:
            model_style = ModelStyle.OSSMODEL

        name, extras = infer_provider(_Handler())
        assert name == "vllm"
        assert extras.get("bfcl.oss.backend") == "vllm"

    def test_oss_sglang(self, monkeypatch):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        monkeypatch.setenv("BFCL_BACKEND", "sglang")

        class _Handler:
            model_style = ModelStyle.OSSMODEL

        name, extras = infer_provider(_Handler())
        assert name == "sglang"

    def test_oss_unknown_backend(self, monkeypatch):
        from bfcl_eval.constants.enums import ModelStyle

        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        monkeypatch.delenv("BFCL_BACKEND", raising=False)

        class _Handler:
            model_style = ModelStyle.OSSMODEL

        name, extras = infer_provider(_Handler())
        assert name == "oss"
        assert extras.get("bfcl.oss.backend") == "unknown"

    def test_no_model_style(self):
        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            pass

        name, extras = infer_provider(_Handler())
        assert name == "unknown"
        assert extras == {}

    def test_none_model_style(self):
        from opentelemetry.instrumentation.bfclv4.internal.provider import (
            infer_provider,
        )

        class _Handler:
            model_style = None

        name, extras = infer_provider(_Handler())
        assert name == "unknown"
