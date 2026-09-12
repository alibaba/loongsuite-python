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

"""
OpenTelemetry slop-code-bench Instrumentation

Instruments the slop-code benchmark orchestrator lifecycle:
- ENTRY: run_agent (CLI entrypoint)
- CHAIN/workflow: run_agent_on_problem (per-problem)
- TASK: AgentRunner._run_checkpoint (per-checkpoint)
- AGENT: Agent.run_checkpoint (concrete agent invocation)
- STEP: MiniSWEAgent.agent_step (ReAct iteration)
- LLM: grade_file_async (Rubric Judge)
"""

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.slop_code.package import _instruments
from opentelemetry.instrumentation.slop_code.version import __version__
from opentelemetry.instrumentation.slop_code.wrappers.agent import (
    _AgentRunCheckpointWrapper,
)
from opentelemetry.instrumentation.slop_code.wrappers.entry import (
    _EntryWrapper,
    _RunnerEntryWrapper,
)
from opentelemetry.instrumentation.slop_code.wrappers.llm import (
    _RubricGradeWrapper,
)
from opentelemetry.instrumentation.slop_code.wrappers.step import (
    _MiniSWEObservationWrapper,
    _MiniSWEStepWrapper,
)
from opentelemetry.instrumentation.slop_code.wrappers.task import (
    _TaskRunCheckpointWrapper,
)
from opentelemetry.instrumentation.slop_code.wrappers.tool import (
    _ToolExecuteActionWrapper,
)
from opentelemetry.instrumentation.slop_code.wrappers.workflow import (
    _WorkflowWrapper,
)
from opentelemetry.instrumentation.utils import unwrap

logger = logging.getLogger(__name__)

__all__ = ["SlopCodeInstrumentor", "__version__"]

_MODULE_ENTRY = "slop_code.entrypoints.commands.run_agent"
_MODULE_WORKER = "slop_code.entrypoints.problem_runner.worker"
# slop_code.entrypoints.problem_runner.driver re-imports
# `run_agent_on_problem` via `from .worker import run_agent_on_problem`
# at package-load time, capturing the original function reference. Because
# our wrap happens after that bind, we must additionally replace the local
# binding inside `driver` itself, otherwise the worker subprocess still
# calls the un-wrapped original and the CHAIN span never fires.
_MODULE_DRIVER = "slop_code.entrypoints.problem_runner.driver"
_MODULE_RUNNER = "slop_code.agent_runner.runner"
_MODULE_AGENT = "slop_code.agent_runner.agent"
_MODULE_MINISWE = "slop_code.agent_runner.agents._miniswe_agent"
_MODULE_RUBRIC = "slop_code.metrics.rubric.router"


class SlopCodeInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for slop-code-bench framework."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__,
            __version__,
            tracer_provider=tracer_provider,
        )

        # 3.1 ENTRY span: run_agent
        try:
            wrap_function_wrapper(
                module=_MODULE_ENTRY,
                name="run_agent",
                wrapper=_EntryWrapper(tracer),
            )
        except Exception as e:
            logger.warning(f"Could not wrap run_agent: {e}")

        # 3.2 CHAIN span: run_agent_on_problem
        workflow_wrapper = _WorkflowWrapper(tracer)
        try:
            wrap_function_wrapper(
                module=_MODULE_WORKER,
                name="run_agent_on_problem",
                wrapper=workflow_wrapper,
            )
        except Exception as e:
            logger.warning(f"Could not wrap run_agent_on_problem: {e}")
        # Also wrap the re-bound name inside driver. driver.py imports
        # run_agent_on_problem at module-load time via `from .worker import ...`,
        # so the local name escapes our worker-module patch. The worker
        # subprocess inherits this stale reference via fork(), and CHAIN
        # spans never fire unless we patch the local re-bind too.
        try:
            wrap_function_wrapper(
                module=_MODULE_DRIVER,
                name="run_agent_on_problem",
                wrapper=workflow_wrapper,
            )
        except Exception as e:
            logger.warning(f"Could not wrap driver.run_agent_on_problem: {e}")

        # 3.3 ENTRY span inside worker: AgentRunner.run
        try:
            wrap_function_wrapper(
                module=_MODULE_RUNNER,
                name="AgentRunner.run",
                wrapper=_RunnerEntryWrapper(tracer),
            )
        except Exception as e:
            logger.warning(f"Could not wrap AgentRunner.run: {e}")

        # 3.4 TASK span: AgentRunner._run_checkpoint
        try:
            wrap_function_wrapper(
                module=_MODULE_RUNNER,
                name="AgentRunner._run_checkpoint",
                wrapper=_TaskRunCheckpointWrapper(tracer),
            )
        except Exception as e:
            logger.warning(f"Could not wrap AgentRunner._run_checkpoint: {e}")

        # 3.5 AGENT span: Agent.run_checkpoint
        try:
            wrap_function_wrapper(
                module=_MODULE_AGENT,
                name="Agent.run_checkpoint",
                wrapper=_AgentRunCheckpointWrapper(tracer),
            )
        except Exception as e:
            logger.warning(f"Could not wrap Agent.run_checkpoint: {e}")

        # 3.6 STEP span: MiniSWEAgent.agent_step
        try:
            wrap_function_wrapper(
                module=_MODULE_MINISWE,
                name="MiniSWEAgent.agent_step",
                wrapper=_MiniSWEStepWrapper(tracer),
            )
        except Exception as e:
            logger.debug(f"Could not wrap MiniSWEAgent.agent_step: {e}")

        # 3.6 STEP end: MiniSWEAgent.get_observation
        try:
            wrap_function_wrapper(
                module=_MODULE_MINISWE,
                name="MiniSWEAgent.get_observation",
                wrapper=_MiniSWEObservationWrapper(tracer),
            )
        except Exception as e:
            logger.debug(f"Could not wrap MiniSWEAgent.get_observation: {e}")

        # 3.7 TOOL span: MiniSWEAgent.execute_action
        try:
            wrap_function_wrapper(
                module=_MODULE_MINISWE,
                name="MiniSWEAgent.execute_action",
                wrapper=_ToolExecuteActionWrapper(tracer),
            )
        except Exception as e:
            logger.debug(f"Could not wrap MiniSWEAgent.execute_action: {e}")

        # 3.8 LLM span: grade_file_async
        try:
            wrap_function_wrapper(
                module=_MODULE_RUBRIC,
                name="grade_file_async",
                wrapper=_RubricGradeWrapper(tracer),
            )
        except Exception as e:
            logger.debug(f"Could not wrap grade_file_async: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            import slop_code.entrypoints.commands.run_agent as mod_entry

            unwrap(mod_entry, "run_agent")
        except Exception:
            pass

        try:
            import slop_code.entrypoints.problem_runner.worker as mod_worker

            unwrap(mod_worker, "run_agent_on_problem")
        except Exception:
            pass

        try:
            import slop_code.entrypoints.problem_runner.driver as mod_driver

            unwrap(mod_driver, "run_agent_on_problem")
        except Exception:
            pass

        try:
            import slop_code.agent_runner.runner as mod_runner

            unwrap(mod_runner.AgentRunner, "_run_checkpoint")
        except Exception:
            pass

        try:
            import slop_code.agent_runner.agent as mod_agent

            unwrap(mod_agent.Agent, "run_checkpoint")
        except Exception:
            pass

        try:
            import slop_code.agent_runner.agents.miniswe as mod_miniswe

            unwrap(mod_miniswe.MiniSWEAgent, "agent_step")
        except Exception:
            pass

        try:
            import slop_code.metrics.rubric.router as mod_rubric

            unwrap(mod_rubric, "grade_file_async")
        except Exception:
            pass
