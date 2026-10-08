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

"""Business calls stay outside advice; tokens never cross a yield boundary."""

from collections.abc import AsyncGenerator

from . import _telemetry as telemetry


class Stream(AsyncGenerator):
    def __init__(self, stream, state):
        self._stream = stream
        self._state = state
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._advance(self._stream.__anext__)

    async def asend(self, value):
        return await self._advance(self._stream.asend, value)

    async def athrow(self, *args):
        return await self._advance(self._stream.athrow, *args)

    async def _advance(self, advance, *args):
        token = telemetry.attach(self._state)
        try:
            try:
                value = await advance(*args)
            except StopAsyncIteration:
                telemetry.finish(self._state)
                raise
            except BaseException as exc:
                telemetry.error(self._state, exc)
                telemetry.finish(self._state)
                raise
            telemetry.response(self._state, value)
            if telemetry.is_terminal(value):
                telemetry.finish(self._state)
            return value
        finally:
            telemetry.detach(token)

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        token = telemetry.attach(self._state)
        try:
            close = getattr(self._stream, "aclose", None)
            if close is not None:
                await close()
        finally:
            telemetry.detach(token)
            telemetry.finish(self._state)

    def __del__(self):
        # Do not attach/detach tokens or execute application cleanup in a finalizer.
        telemetry.finish(self._state)


def wrap(tracer, method, server, streaming):
    if streaming:

        def stream_wrapper(wrapped, instance, args, kwargs):
            state = telemetry.start(
                tracer, method, instance, args, kwargs, server
            )
            token = telemetry.attach(state)
            try:
                try:
                    stream = wrapped(*args, **kwargs)
                except BaseException as exc:
                    telemetry.error(state, exc)
                    telemetry.finish(state)
                    raise
            finally:
                telemetry.detach(token)
            if state is None:
                return stream
            try:
                return Stream(stream, state)
            except Exception:  # A proxy construction failure must not change the business result.
                telemetry.finish(state)
                return stream

        return stream_wrapper

    async def coroutine_wrapper(wrapped, instance, args, kwargs):
        state = telemetry.start(tracer, method, instance, args, kwargs, server)
        token = telemetry.attach(state)
        try:
            try:
                result = await wrapped(*args, **kwargs)
            except BaseException as exc:
                telemetry.error(state, exc)
                raise
            telemetry.response(state, result)
            return result
        finally:
            telemetry.detach(token)
            telemetry.finish(state)

    return coroutine_wrapper


async def send(wrapped, instance, args, kwargs):
    request = args[0] if args else kwargs.get("request")
    telemetry.inject(request)
    return await wrapped(*args, **kwargs)


async def request(wrapped, instance, args, kwargs):
    token = telemetry.request_headers(
        args[0] if args else kwargs.get("request")
    )
    try:
        return await wrapped(*args, **kwargs)
    finally:
        telemetry.detach(token)
