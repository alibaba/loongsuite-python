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

"""Context-propagating ``ThreadPoolExecutor`` used by the ENTRY wrapper.

``concurrent.futures.ThreadPoolExecutor`` does not automatically copy the
current ``contextvars`` context (which holds the OTel current span) into
worker threads.  We subclass it and copy ``contextvars.copy_context()`` per
``submit`` so the AGENT span created inside the worker thread can attach as
a child of the ENTRY span.

We only swap the ``ThreadPoolExecutor`` *name* in the
``bfcl_eval._llm_response_generation`` namespace; the global
``concurrent.futures.ThreadPoolExecutor`` is untouched.
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor as _RealExecutor


class ContextPropagatingExecutor(_RealExecutor):
    """``ThreadPoolExecutor`` that propagates the calling ``Context``.

    Only the ``submit`` method is overridden because BFCL only uses
    ``submit`` (see ``_llm_response_generation.generate_results``).
    """

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        ctx = contextvars.copy_context()
        return super().submit(ctx.run, fn, *args, **kwargs)
