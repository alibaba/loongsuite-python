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

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest.mock
from typing import Any


def create_mock_model(**fields: Any) -> Any:
    """Mock a pydantic model of the Interactions API with these fields.

    Since google-genai 2.34, ``interactions.create`` copies the interaction
    it returns, and its usage, with ``model_copy(update=...)`` (to merge
    automatic continuation hops, even for a single hop). On a plain
    ``MagicMock`` that copy would be a new mock without these fields.
    """
    mock = unittest.mock.MagicMock()
    mock.configure_mock(**fields)

    def model_copy(update: Any = None, **kwargs: Any) -> Any:
        return create_mock_model(**{**fields, **(update or {})})

    mock.model_copy.side_effect = model_copy
    return mock


def create_mock_interaction(
    interaction_id: str = "test-id",
    model_name: str = "test-model",
    input_text: str = "user input",
    output_text: str = "model output",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> Any:
    mock_usage = create_mock_model(
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_thought_tokens=0,
        total_cached_tokens=0,
    )

    mock_user_step = unittest.mock.MagicMock()
    mock_user_step.type = "user_input"
    mock_user_part = unittest.mock.MagicMock()
    mock_user_part.text = input_text
    mock_user_step.content = [mock_user_part]

    mock_model_step = unittest.mock.MagicMock()
    mock_model_step.type = "model_output"
    mock_model_part = unittest.mock.MagicMock()
    mock_model_part.text = output_text
    mock_model_step.content = [mock_model_part]

    return create_mock_model(
        id=interaction_id,
        model=model_name,
        usage=mock_usage,
        steps=[mock_user_step, mock_model_step],
        output_text=output_text,
    )


def create_mock_completed_event(interaction: Any) -> Any:
    event = unittest.mock.MagicMock()
    event.event_type = "interaction_completed"
    event.interaction = interaction
    return event
