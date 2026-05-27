# LoongSuite CrewAI Instrumentation

This library provides automatic instrumentation for [CrewAI](https://www.crewai.com/), a framework for orchestrating role-playing, autonomous AI agents.

## Installation

```bash
# Step 1: install LoongSuite distro
pip install loongsuite-distro

# Step 2 (Option C): install this instrumentation from PyPI
pip install loongsuite-instrumentation-crewai
```

## Usage

```python
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor

# Instrument CrewAI
CrewAIInstrumentor().instrument()

# Your CrewAI code here
from crewai import Agent, Task, Crew

agent = Agent(
    role='Data Analyst',
    goal='Extract actionable insights',
    backstory='Expert in data analysis',
    verbose=True
)

task = Task(
    description='Analyze the latest AI trends',
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

result = crew.kickoff()
```

## Supported Versions

- CrewAI >= 0.80.0

## References

- [CrewAI Documentation](https://docs.crewai.com/)
- [OpenTelemetry Python Documentation](https://opentelemetry-python.readthedocs.io/)

