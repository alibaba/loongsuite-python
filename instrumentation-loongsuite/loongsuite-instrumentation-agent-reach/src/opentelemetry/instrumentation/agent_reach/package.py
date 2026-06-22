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

from typing import Sequence

# agent-reach 1.5.0 is the version validated during investigation; newer
# 1.x releases are expected to stay compatible because all patched entry
# points are stable (cli.main, doctor.check_all, Channel.check,
# probe.probe_command, transcribe, transcribe_chunk, mcp_server.create_server).
_instruments: Sequence[str] = ("agent-reach >= 1.5.0, < 2.0.0",)
