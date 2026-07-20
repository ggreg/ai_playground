"""The Agent — a model in a loop with tools and memory.

This file is intentionally short. The whole "agent" abstraction is essentially:

    while not done:
        response = llm.complete(messages, tools)
        if response.tool_calls:
            for call in response.tool_calls:
                result = registry.dispatch(call)
                messages.append(tool_result)
        else:
            done = True

Everything else (planning, multi-agent, reflection) is variations on this loop.
If you take one thing away from this module, take that loop.

The pattern is called **ReAct** (Yao et al., 2022): the model alternates
between Reasoning (text it generates) and Acting (tool calls). Reasoning
makes the next tool choice better; tool results constrain the next round of
reasoning. That alternation is what makes agents work — pure LLM reasoning
hallucinates, pure tool calls have no glue.

References:
- ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)
  https://arxiv.org/abs/2210.03629
- Anthropic on agent loops: https://www.anthropic.com/research/building-effective-agents
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .llm import LLMBackend, LLMResponse, ToolCall
from .memory import ConversationMemory, VectorMemory
from .tools import ToolRegistry, ToolResult


@dataclass
class AgentStep:
    """One iteration of the agent loop, captured for tracing.

    Inspecting a trace is the single best way to debug an agent — most
    failures are "the model called the wrong tool" or "the tool returned
    junk and the model trusted it", and you'll only see that in the trace.
    """

    iteration: int
    response: LLMResponse
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class AgentResult:
    final_text: str
    steps: list[AgentStep]
    stop_reason: str  # "end_turn" | "max_iterations" | "stop_sequence"


class Agent:
    """A ReAct-style agent.

    Wire it up with an LLM backend, some tools, and (optionally) a memory.
    Then call `.run(user_message)` to do one task end-to-end.

        agent = Agent(
            llm=ClaudeBackend(),
            tools=ToolRegistry(builtin_tools()),
            system="You are a helpful research assistant.",
        )
        result = agent.run("Find the population of Tokyo and divide by 3.")
        print(result.final_text)
    """

    def __init__(
        self,
        llm: LLMBackend,
        tools: ToolRegistry | None = None,
        system: str | None = None,
        memory: ConversationMemory | None = None,
        long_term: VectorMemory | None = None,
        max_iterations: int = 10,
        max_tokens: int = 1024,
        on_step: Any = None,  # optional callback(AgentStep) for streaming/printing
    ):
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.system = system
        self.memory = memory or ConversationMemory()
        self.long_term = long_term
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.on_step = on_step

    def run(self, user_message: str) -> AgentResult:
        """Execute one task. Loops until the model stops asking for tools.

        The loop has three termination conditions:
        1. Model returns no tool calls (`stop_reason == "end_turn"`) — done.
        2. We hit `max_iterations` — safety belt for runaway agents.
        3. A tool raises — we report the error to the model, which can retry
           or give up; we do *not* terminate.
        """
        self._inject_long_term_context(user_message)
        self.memory.add(role="user", content=user_message)

        steps: list[AgentStep] = []
        for i in range(self.max_iterations):
            response = self.llm.complete(
                messages=self.memory.messages(),
                tools=self.tools.list() or None,
                system=self.system,
                max_tokens=self.max_tokens,
            )

            # Record the assistant turn (text + tool_use blocks) in memory.
            # Anthropic's API requires the assistant message that *requested*
            # tool calls to be present before the tool_result message.
            self.memory.add(
                role="assistant",
                content=self._assistant_blocks(response),
            )

            step = AgentStep(iteration=i, response=response)

            if not response.tool_calls:
                steps.append(step)
                if self.on_step:
                    self.on_step(step)
                return AgentResult(
                    final_text=response.text,
                    steps=steps,
                    stop_reason=response.stop_reason,
                )

            # Dispatch every tool call the model made this turn (it can make
            # several in parallel) and feed them back as a single user message.
            tool_results = [
                self.tools.dispatch(call.name, call.arguments, call.id)
                for call in response.tool_calls
            ]
            step.tool_results = tool_results
            steps.append(step)
            if self.on_step:
                self.on_step(step)

            self.memory.add(
                role="user",
                content=[r.to_anthropic_block() for r in tool_results],
            )

        # Fell out of the loop without a clean end_turn.
        return AgentResult(
            final_text=steps[-1].response.text if steps else "",
            steps=steps,
            stop_reason="max_iterations",
        )

    def _assistant_blocks(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Reconstruct the assistant message as content blocks.

        Anthropic's API needs the exact original blocks back when continuing
        a tool-use turn. Text first, then tool_use blocks in the order the
        model emitted them.
        """
        blocks: list[dict[str, Any]] = []
        if response.text:
            blocks.append({"type": "text", "text": response.text})
        for call in response.tool_calls:
            blocks.append({
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            })
        return blocks or [{"type": "text", "text": ""}]

    def _inject_long_term_context(self, user_message: str) -> None:
        """If we have long-term memory, retrieve and inject the top hits.

        This is the simplest possible RAG pattern: every user message
        triggers a similarity search; matches get prepended as a system note.
        Real systems are pickier (only retrieve when the model asks, or only
        for messages that look like questions) — but this works surprisingly
        well as a baseline.
        """
        if self.long_term is None or len(self.long_term) == 0:
            return
        hits = self.long_term.search(user_message, top_k=3)
        if not hits:
            return
        recall = "\n".join(f"- ({score:.2f}) {entry.text}" for score, entry in hits)
        self.memory.add(
            role="user",
            content=f"[relevant memories]\n{recall}",
        )


__all__ = ["Agent", "AgentResult", "AgentStep", "ToolCall"]
