"""p6 — your ReAct agent loop. See README.md for the brief and rules (short version:
the repo's agents package is the after-the-fact comparison, not an import).
"""

from typing import Callable

from scripted_llm import ScriptedLLM  # noqa: F401  (any object with .complete works)


class Agent:
    """A ReAct loop: model -> parse -> tool -> observation -> model, until FINAL.

    Pinned protocol (the tests hold you to exactly this):
    - Messages are dicts {"role": ..., "content": ...}. The task enters as
      {"role": "user", "content": task}; each model reply is appended as role
      "assistant"; each observation as {"role": "user", "content": "OBSERVATION: ..."}.
    - The model replies with either
        FINAL: <answer>                      -> run() returns "<answer>" (stripped)
      or a line
        ACTION: <tool_name>(<json object>)   -> call tools[tool_name](**parsed_json),
      append "OBSERVATION: <repr of return value>", and loop.
    - Unknown tool, unparseable action, or a tool that raises: append
      "OBSERVATION: ERROR: <description>" (naming the offending tool / carrying the
      exception message) and keep looping — a broken tool is information for the
      model, not a crash of the loop.
    - At most max_steps model calls; if no FINAL by then, return "(gave up)".
    """

    def __init__(self, llm, tools: dict[str, Callable], max_steps: int = 8):
        raise NotImplementedError

    def run(self, task: str) -> str:
        raise NotImplementedError
