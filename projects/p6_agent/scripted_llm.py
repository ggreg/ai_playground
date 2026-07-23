"""p6 scaffolding — a deterministic stand-in for an LLM API. Read it, don't edit it.

ScriptedLLM replays canned responses in order and records every `messages` list it was
called with (in `.calls`), so tests can verify what your loop actually showed the model
— the transcript IS the agent's behavior.
"""


class ScriptedLLM:
    """Duck-types an LLM backend: complete(messages) -> str, from a fixed script."""

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> str:
        # Record a snapshot: the caller may keep mutating its messages list.
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            raise AssertionError(
                "ScriptedLLM ran out of responses — the loop called the model more "
                "times than the test's script allows"
            )
        return self.script.pop(0)
