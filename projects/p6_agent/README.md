# p6 — An agent from a blank file

**After Phase 6 (Building AI Agents).** You've traced the repo's `Agent` loop. Now write
your own ReAct loop ([Yao et al., 2022](https://arxiv.org/abs/2210.03629), docs/PAPERS.md)
— parse, dispatch, observe, repeat — against a **scripted** LLM, so every test is
deterministic and no API key is ever needed. ~2–3 hours.

## You build (the contract in `agent_scratch.py`)

- `Agent(llm, tools, max_steps)` with `run(task) -> str`. Per turn: call
  `llm.complete(messages)`, parse the reply (`FINAL:` or `ACTION: name({...json...})`),
  dispatch to the tool, append the observation, loop. The exact message and observation
  formats are pinned in the stub's docstring — the ScriptedLLM records every `messages`
  list it receives, and the tests inspect that transcript.

The tests exercise the three things that separate a loop from a demo: a correct
happy-path transcript, **tool failure fed back as an observation** (not a crash), and a
**step budget** that actually halts a model that never says FINAL.

## Scaffolding provided

- `scripted_llm.py` — `ScriptedLLM`: replays a canned list of responses and records every
  prompt. Faking an LLM deterministically is boilerplate; deciding what your loop does
  with its replies is the project.

## Rules

- No imports from `ai_playground.agents`, and no real API calls. Everything else
  (including `json` and `re`) is fair game.

## Done-when

```bash
uv run pytest projects/p6_agent/ -v
```

Afterwards: diff against `src/ai_playground/agents/agent.py` (<100 lines — how close did
you land?), then wire your loop to `ClaudeBackend` and give it a real tool.
