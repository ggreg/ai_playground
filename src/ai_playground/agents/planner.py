"""Plan-and-Execute — decompose first, then act.

ReAct is "think one step ahead at a time". That's fine for simple tasks but
breaks down on multi-step problems where the model loses the thread halfway
through. Plan-and-Execute splits the job:

1. **Planner**: read the goal, emit an ordered list of steps. Pure reasoning,
   no tools.
2. **Executor**: an Agent runs each step in turn, with tools.
3. **Reflector** (optional): after each step, the planner checks whether the
   plan still makes sense; if not, replan.

When this beats ReAct: long horizons (5+ steps), goals where the right tool
sequence is non-obvious. When ReAct wins: short tasks (the planner overhead
is wasted), and tasks where intermediate results meaningfully change strategy
(ReAct can adapt every turn, this only adapts on replans).

References:
- Plan-and-Solve Prompting (Wang et al., 2023): https://arxiv.org/abs/2305.04091
- LLM+P (Liu et al., 2023): https://arxiv.org/abs/2304.11477
- Reflexion (Shinn et al., 2023): https://arxiv.org/abs/2303.11366
  — the reflection-after-failure pattern this module supports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .agent import Agent, AgentResult
from .llm import LLMBackend
from .tools import ToolRegistry


@dataclass
class Plan:
    goal: str
    steps: list[str]


@dataclass
class PlanExecutionResult:
    plan: Plan
    step_results: list[AgentResult] = field(default_factory=list)
    final_summary: str = ""


PLANNER_PROMPT = """You are a planning module. Given a user's goal, output a JSON
array of 2-6 concrete steps that, executed in order, accomplish the goal. Each
step should be one specific action that an agent with tools (file I/O, shell,
web fetch, calculator) can do in a single turn.

Output ONLY the JSON array, no commentary. Example:

  ["Fetch the NASA homepage", "Extract the headline article URL",
   "Fetch that article and summarize its key points"]

Goal: {goal}"""


class Planner:
    """Decompose a goal into steps with one LLM call."""

    def __init__(self, llm: LLMBackend, max_tokens: int = 512):
        self.llm = llm
        self.max_tokens = max_tokens

    def plan(self, goal: str) -> Plan:
        response = self.llm.complete(
            messages=[{"role": "user", "content": PLANNER_PROMPT.format(goal=goal)}],
            max_tokens=self.max_tokens,
        )
        steps = self._parse_steps(response.text)
        return Plan(goal=goal, steps=steps)

    @staticmethod
    def _parse_steps(text: str) -> list[str]:
        # Models love to wrap JSON in ```json fences — strip them.
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: split on newlines, drop empties and numbering.
            lines = [line.lstrip("-0123456789. ").strip() for line in text.splitlines()]
            return [line for line in lines if line]
        if isinstance(data, list):
            return [str(s) for s in data]
        return [str(data)]


class PlanAndExecute:
    """Run a planner, then drive an Agent through each step.

    The same Agent (and its memory) handles every step, so later steps can
    refer back to earlier results. If you want isolated steps, instantiate a
    fresh Agent per step instead — depends on the task.
    """

    def __init__(
        self,
        llm: LLMBackend,
        tools: ToolRegistry,
        system: str | None = None,
        reflect: bool = False,
    ):
        self.planner = Planner(llm)
        self.executor = Agent(llm=llm, tools=tools, system=system)
        self.llm = llm
        self.reflect = reflect

    def run(self, goal: str) -> PlanExecutionResult:
        plan = self.planner.plan(goal)
        result = PlanExecutionResult(plan=plan)

        for idx, step in enumerate(plan.steps):
            step_instruction = (
                f"Goal: {goal}\n"
                f"Step {idx + 1} of {len(plan.steps)}: {step}\n"
                f"Do this step using tools as needed, then report what you did and learned."
            )
            step_result = self.executor.run(step_instruction)
            result.step_results.append(step_result)

            if self.reflect and idx < len(plan.steps) - 1:
                # Ask the planner whether to revise the remaining steps.
                remaining = plan.steps[idx + 1 :]
                revised = self._maybe_replan(goal, plan.steps[: idx + 1], step_result, remaining)
                if revised is not None:
                    plan.steps = plan.steps[: idx + 1] + revised

        # One final summary turn — what was the answer?
        summary = self.executor.run(
            f"You've completed the plan for: {goal!r}. Summarize the final answer in 1-3 sentences."
        )
        result.final_summary = summary.final_text
        return result

    def _maybe_replan(
        self,
        goal: str,
        done: list[str],
        last_result: AgentResult,
        remaining: list[str],
    ) -> list[str] | None:
        prompt = (
            f"Goal: {goal}\n"
            f"Steps completed: {done}\n"
            f"Last step output: {last_result.final_text[:500]}\n"
            f"Remaining steps: {remaining}\n\n"
            "Are the remaining steps still the best path to the goal? "
            "If yes, reply with the single word KEEP. "
            "If no, output a new JSON array replacing the remaining steps."
        )
        response = self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        if "KEEP" in response.text.upper()[:20]:
            return None
        return Planner._parse_steps(response.text)


__all__ = ["Plan", "PlanAndExecute", "PlanExecutionResult", "Planner"]
