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

import os

import pytest


def pytest_configure(config: pytest.Config):
    # 尝试获取环境变量
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if api_key is None:
        pytest.exit(
            "Environment variable 'DEEPSEEK_API_KEY' is not set. Aborting tests."
        )
    else:
        # 将环境变量保存到全局配置中，以便后续测试使用
        config.option.api_key = api_key
