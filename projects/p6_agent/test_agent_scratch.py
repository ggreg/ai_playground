"""Acceptance tests for p6 — the agent loop.

The ScriptedLLM records every messages list it receives; the transcript is what's
verified — what the model was shown matters as much as what run() returns.
"""

from pathlib import Path


from agent_scratch import Agent
from scripted_llm import ScriptedLLM


def _tools():
    return {
        "calc": lambda expr: eval(expr, {"__builtins__": {}}),  # noqa: S307 — toy sandbox
        "broken": lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    }


def test_no_forbidden_imports():
    lines = (Path(__file__).parent / "agent_scratch.py").read_text().splitlines()
    code = "\n".join(ln for ln in lines if not ln.lstrip().startswith("#"))
    assert "ai_playground.agents" not in code, "build it yourself — see README rules"


def test_happy_path_tool_call(attempt):
    llm = ScriptedLLM([
        'ACTION: calc({"expr": "12 * 7"})',
        "FINAL: The answer is 84",
    ])
    agent = attempt(Agent, llm, _tools(), 8)
    result = attempt(agent.run, "What is 12 times 7?")

    assert result == "The answer is 84"
    assert len(llm.calls) == 2
    first, second = llm.calls
    assert any(m["role"] == "user" and "12 times 7" in m["content"] for m in first), (
        "the task must be in the first messages list"
    )
    assert any(
        m["content"].startswith("OBSERVATION:") and "84" in m["content"] for m in second
    ), f"turn 2 must contain 'OBSERVATION: 84', got: {[m['content'] for m in second]}"
    assert any(m["role"] == "assistant" for m in second), (
        "the model's own ACTION reply must be part of the history it sees next turn"
    )


def test_tool_error_becomes_observation(attempt):
    llm = ScriptedLLM([
        'ACTION: broken({})',
        'ACTION: no_such_tool({"x": 1})',
        "FINAL: done",
    ])
    agent = attempt(Agent, llm, _tools(), 8)
    result = attempt(agent.run, "try the tools")

    assert result == "done", "tool failures must not crash or end the loop"
    assert len(llm.calls) == 3
    for turn, needle in ((llm.calls[1], "boom"), (llm.calls[2], "no_such_tool")):
        obs = [m["content"] for m in turn if m["content"].startswith("OBSERVATION: ERROR")]
        assert obs and any(needle in o for o in obs), (
            f"expected an 'OBSERVATION: ERROR: ... {needle} ...' message, "
            f"got: {[m['content'] for m in turn]}"
        )


def test_step_budget_halts_the_loop(attempt):
    llm = ScriptedLLM(['ACTION: calc({"expr": "1 + 1"})'] * 10)
    agent = attempt(Agent, llm, _tools(), 3)
    result = attempt(agent.run, "loop forever")

    assert isinstance(result, str)
    assert len(llm.calls) == 3, (
        f"max_steps=3 must mean at most 3 model calls, got {len(llm.calls)}"
    )


def test_multi_step_state_accumulates(attempt):
    llm = ScriptedLLM([
        'ACTION: calc({"expr": "6 * 7"})',
        'ACTION: calc({"expr": "42 + 58"})',
        "FINAL: 100",
    ])
    agent = attempt(Agent, llm, _tools(), 8)
    assert attempt(agent.run, "chain two calculations") == "100"
    last = llm.calls[-1]
    contents = [m["content"] for m in last]
    assert any("42" in c and c.startswith("OBSERVATION") for c in contents)
    assert any("100" in c and c.startswith("OBSERVATION") for c in contents), (
        "every earlier observation must still be visible in the final turn's messages"
    )
