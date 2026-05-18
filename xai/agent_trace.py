"""
Agent decision-step logging for tool-call-style explainability.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional
import time


@dataclass
class AgentDecisionStep:
    step_id: int
    tool_name: str
    tool_description: str
    input_summary: str
    output_summary: str
    rationale: str
    status: str = "completed"
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRunTrace:
    user_request: str
    steps: list[AgentDecisionStep] = field(default_factory=list)
    final_output: str = ""

    def add_step(self, step: AgentDecisionStep) -> None:
        self.steps.append(step)

    def to_context(self) -> str:
        lines = ["AGENT DECISION TRACE (tool calls):", ""]
        for s in self.steps:
            lines.append(f"Step {s.step_id}: {s.tool_name}")
            lines.append(f"  Tool: {s.tool_description}")
            lines.append(f"  Input: {s.input_summary}")
            lines.append(f"  Output: {s.output_summary}")
            lines.append(f"  Why: {s.rationale}")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_request": self.user_request,
            "steps": [s.to_dict() for s in self.steps],
            "final_output": self.final_output,
        }


class StepTimer:
    def __init__(self):
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000
