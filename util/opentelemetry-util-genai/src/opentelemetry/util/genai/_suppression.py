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

from contextlib import contextmanager
from typing import Generator

from opentelemetry import context
from opentelemetry.context import _SUPPRESS_HTTP_INSTRUMENTATION_KEY


@contextmanager
def suppress_http_instrumentation() -> Generator[None, None, None]:
    """Suppress internal HTTP requests using the key HTTP instrumentors share.

    This private Context key is the compatibility boundary: creating a new key
    with the same name would not be recognized by existing instrumentors.
    Robin packaging rewrites both imports to its commercial Context namespace.
    """
    token = context.attach(
        context.set_value(_SUPPRESS_HTTP_INSTRUMENTATION_KEY, True)
    )
    try:
        yield
    finally:
        context.detach(token)
