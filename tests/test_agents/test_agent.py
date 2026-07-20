"""Tests for the core Agent loop, using a scripted mock LLM.

We don't hit the Anthropic API in tests — that would make the suite slow,
flaky, and require credentials. Instead we drive the loop with a fake backend
that returns pre-baked responses in order. This is also how you'd debug a
suspected agent-loop bug: pin the model's outputs, vary the loop, see what
breaks.
"""

from ai_playground.agents.agent import Agent
from ai_playground.agents.llm import LLMResponse, ToolCall
from ai_playground.agents.tools import Tool, ToolRegistry


class ScriptedBackend:
    """An LLMBackend that returns responses from a fixed script."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages, tools=None, system=None, max_tokens=1024):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        return self.responses.pop(0)


class TestAgentLoop:
    def test_no_tool_calls_terminates_immediately(self):
        backend = ScriptedBackend([LLMResponse(text="hi back", stop_reason="end_turn")])
        agent = Agent(llm=backend)
        result = agent.run("hi")
        assert result.final_text == "hi back"
        assert result.stop_reason == "end_turn"
        assert len(result.steps) == 1

    def test_single_tool_call_then_answer(self):
        adder = Tool(
            name="add",
            description="add two ints",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            func=lambda a, b: a + b,
        )
        backend = ScriptedBackend([
            LLMResponse(
                text="I'll use the calculator.",
                tool_calls=[ToolCall(id="t1", name="add", arguments={"a": 40, "b": 2})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="The answer is 42.", stop_reason="end_turn"),
        ])
        agent = Agent(llm=backend, tools=ToolRegistry([adder]))
        result = agent.run("what is 40 + 2?")

        assert result.final_text == "The answer is 42."
        assert result.stop_reason == "end_turn"
        assert len(result.steps) == 2
        assert result.steps[0].tool_results[0].content == "42"
        assert result.steps[0].tool_results[0].is_error is False

    def test_tool_error_is_reported_not_raised(self):
        broken = Tool(name="broken", description="", input_schema={}, func=lambda: 1 / 0)
        backend = ScriptedBackend([
            LLMResponse(
                tool_calls=[ToolCall(id="t1", name="broken", arguments={})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="That failed; giving up.", stop_reason="end_turn"),
        ])
        agent = Agent(llm=backend, tools=ToolRegistry([broken]))
        result = agent.run("try the broken tool")

        # Agent should keep going after a tool error, not crash.
        assert result.final_text == "That failed; giving up."
        assert result.steps[0].tool_results[0].is_error is True

    def test_parallel_tool_calls_in_one_turn(self):
        echo = Tool(
            name="echo",
            description="",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
            func=lambda x: x.upper(),
        )
        backend = ScriptedBackend([
            LLMResponse(
                tool_calls=[
                    ToolCall(id="a", name="echo", arguments={"x": "foo"}),
                    ToolCall(id="b", name="echo", arguments={"x": "bar"}),
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(text="done", stop_reason="end_turn"),
        ])
        agent = Agent(llm=backend, tools=ToolRegistry([echo]))
        result = agent.run("echo foo and bar")

        results = result.steps[0].tool_results
        assert len(results) == 2
        assert {r.content for r in results} == {"FOO", "BAR"}

    def test_max_iterations_stops_runaway_loop(self):
        # Always asks for a tool, never finishes.
        runaway = LLMResponse(
            tool_calls=[ToolCall(id="x", name="noop", arguments={})],
            stop_reason="tool_use",
        )
        noop = Tool(name="noop", description="", input_schema={}, func=lambda: "ok")

        backend = ScriptedBackend([runaway] * 20)
        agent = Agent(llm=backend, tools=ToolRegistry([noop]), max_iterations=3)
        result = agent.run("loop forever")

        assert result.stop_reason == "max_iterations"
        assert len(result.steps) == 3

    def test_system_prompt_passed_through(self):
        backend = ScriptedBackend([LLMResponse(text="ok", stop_reason="end_turn")])
        agent = Agent(llm=backend, system="you are terse")
        agent.run("hello")
        assert backend.calls[0]["system"] == "you are terse"
