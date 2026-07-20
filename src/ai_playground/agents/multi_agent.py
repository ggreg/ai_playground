"""Multi-agent orchestration — several specialized agents cooperating.

The pattern that keeps showing up: a **Supervisor** routes work to one of N
**Worker** agents, each with its own tools and system prompt. Each worker is
narrowly competent; the supervisor's job is to pick the right one and pass
results between them.

Why this beats one big agent:
- **Focused tool surfaces**. A 30-tool agent gets worse at choosing than a
  3-tool agent. Splitting tools across roles makes each model call easier.
- **Specialized prompts**. The researcher prompt and the writer prompt pull
  the model toward different behaviors; one combined prompt waters both down.
- **Cheap delegation**. The supervisor can be a small model (route decisions
  are easy); workers can be the big one (actual work is hard).

Why this is overkill for most tasks: more LLM calls, more failure modes,
more state. Use it when the roles are *genuinely* distinct, not just because
"more agents" sounds better.

References:
- AutoGen (Wu et al., 2023): https://arxiv.org/abs/2308.08155
  — the canonical "agents-as-conversation" framework.
- Generative Agents (Park et al., 2023): https://arxiv.org/abs/2304.03442
- ChatDev (Qian et al., 2023): https://arxiv.org/abs/2307.07924
  — multi-role software engineering org as agents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .agent import Agent
from .llm import LLMBackend
from .tools import ToolRegistry


@dataclass
class Worker:
    """A named, specialized Agent.

    Keep `description` concrete — the supervisor uses it to route, and vague
    descriptions like "general helper" make routing terrible.
    """

    name: str
    description: str
    agent: Agent


@dataclass
class MultiAgentResult:
    final_text: str
    transcript: list[tuple[str, str]] = field(default_factory=list)  # (worker_name, output)


SUPERVISOR_PROMPT_TEMPLATE = """You are the supervisor of a team of specialized agents.
Your job: read the user's request, decide which worker should handle the next
step, and pass them a clear instruction. After the worker reports back, decide
whether to call another worker or whether the task is done.

Available workers:
{worker_list}

Respond with EITHER:
  {{"action": "delegate", "worker": "<name>", "instruction": "<what they should do>"}}
OR:
  {{"action": "finish", "answer": "<final response to the user>"}}

Output ONLY the JSON object, no surrounding text."""


class Supervisor:
    """Routes tasks to workers in a loop.

    Each turn the supervisor sees the original goal plus every worker
    response so far, and decides: delegate again, or finish. The loop ends
    when the supervisor emits "finish" or we hit `max_steps`.
    """

    def __init__(
        self,
        llm: LLMBackend,
        workers: list[Worker],
        max_steps: int = 8,
    ):
        self.llm = llm
        self.workers = {w.name: w for w in workers}
        self.max_steps = max_steps
        self._worker_list = "\n".join(f"- {w.name}: {w.description}" for w in workers)

    def run(self, user_request: str) -> MultiAgentResult:
        transcript: list[tuple[str, str]] = []
        history: list[dict[str, str]] = [{"role": "user", "content": user_request}]

        for _ in range(self.max_steps):
            decision = self._decide(history)

            if decision.get("action") == "finish":
                final = decision.get("answer", "")
                transcript.append(("supervisor", final))
                return MultiAgentResult(final_text=final, transcript=transcript)

            worker_name = decision.get("worker")
            instruction = decision.get("instruction", "")
            worker = self.workers.get(worker_name)
            if worker is None:
                # Supervisor named a non-existent worker — surface this and try again.
                history.append({
                    "role": "user",
                    "content": f"No worker named {worker_name!r}. Pick from: {sorted(self.workers)}",
                })
                continue

            result = worker.agent.run(instruction)
            transcript.append((worker.name, result.final_text))
            history.append({
                "role": "assistant",
                "content": json.dumps(decision),
            })
            history.append({
                "role": "user",
                "content": f"[{worker.name} reports]: {result.final_text}",
            })

        # Out of steps — return whatever the last worker said.
        last = transcript[-1][1] if transcript else ""
        return MultiAgentResult(final_text=last, transcript=transcript)

    def _decide(self, history: list[dict[str, str]]) -> dict:
        system = SUPERVISOR_PROMPT_TEMPLATE.format(worker_list=self._worker_list)
        response = self.llm.complete(
            messages=history,
            system=system,
            max_tokens=400,
        )
        text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"action": "finish", "answer": response.text}


def researcher_writer_critic(
    llm: LLMBackend,
    tools: ToolRegistry,
) -> Supervisor:
    """Convenience factory for the classic three-role setup.

    - **researcher**: gathers facts (web_fetch, file_read).
    - **writer**: turns facts into prose.
    - **critic**: reviews and demands revisions.

    Useful for content generation tasks where one-shot output is too
    superficial but you don't need a full custom team.
    """
    workers = [
        Worker(
            name="researcher",
            description="Gathers facts from the web or files. Returns raw notes and sources.",
            agent=Agent(
                llm=llm,
                tools=tools,
                system="You are a research assistant. Gather facts and return them with sources. Be terse.",
            ),
        ),
        Worker(
            name="writer",
            description="Takes notes from the researcher and produces well-structured prose.",
            agent=Agent(
                llm=llm,
                tools=ToolRegistry(),  # writer doesn't need tools
                system="You are a technical writer. Turn raw notes into clear, well-organized prose.",
            ),
        ),
        Worker(
            name="critic",
            description="Reviews drafts and identifies weaknesses, factual errors, or omissions.",
            agent=Agent(
                llm=llm,
                tools=ToolRegistry(),
                system="You are a sharp critic. Find weaknesses in the draft. Be specific and concise.",
            ),
        ),
    ]
    return Supervisor(llm=llm, workers=workers)


__all__ = ["MultiAgentResult", "Supervisor", "Worker", "researcher_writer_critic"]
