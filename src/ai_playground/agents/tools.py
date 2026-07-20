"""Tools — the functions the agent can actually call.

A "tool" in agent-speak is just a function with:
1. A name and a JSON-schema description of its arguments — so the model knows
   when and how to call it.
2. A callable implementation — so we can actually run it when the model asks.

That's it. Everything else (registries, dispatchers, sandboxing) is plumbing
around this idea.

Why JSON schema? Because that's the format every model provider has converged
on for "structured function calling" (OpenAI function calling, Anthropic tools,
Gemini function declarations all use it). The model is fine-tuned to emit
arguments that match the schema; the schema is what you give it.

Security note: the `shell_exec` tool runs arbitrary commands. For real agents
behind real users, you want sandboxing (containers, seccomp, restricted PATH).
Here it's available because this is a learning repo run locally — but be aware
that "let the model run shell commands" is a real-world security boundary.

References:
- Anthropic tool use docs: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- OpenAI function calling: https://platform.openai.com/docs/guides/function-calling
- Toolformer (Schick et al., 2023): https://arxiv.org/abs/2302.04761
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Tool:
    """A function the model can call.

    `input_schema` follows JSON Schema and gets passed straight to the API.
    Keep descriptions short and concrete — the model reads them every turn.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    func: Callable[..., Any]

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


@dataclass
class ToolResult:
    """The output of a tool call, packaged for handing back to the model.

    `is_error` tells the model the tool failed, which it can use to retry or
    apologize. Without this, you get models that confidently claim success
    after a tool errored.
    """

    tool_call_id: str
    content: str
    is_error: bool = False

    def to_anthropic_block(self) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error,
        }


class ToolRegistry:
    """A name → Tool lookup. The agent calls `dispatch` for each tool_use."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def dispatch(self, name: str, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """Run the tool and capture its output (or its exception).

        We catch all exceptions and turn them into `is_error=True` results so a
        misbehaving tool doesn't kill the whole agent loop. The model will see
        the error message and can decide what to do.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_call_id=call_id,
                content=f"unknown tool: {name!r}. available: {sorted(self._tools)}",
                is_error=True,
            )
        try:
            result = tool(**arguments)
            return ToolResult(tool_call_id=call_id, content=str(result), is_error=False)
        except Exception as e:  # noqa: BLE001 — yes we want to catch everything here
            return ToolResult(
                tool_call_id=call_id,
                content=f"{type(e).__name__}: {e}",
                is_error=True,
            )


# ---------------------------------------------------------------------------
# Builtin tools
# ---------------------------------------------------------------------------
#
# A small starter set. These are deliberately *capability* primitives, not
# task-specific. Real agent setups end up with 5–20 tools chosen for the
# domain (search a wiki, send a Slack message, query a database).


def _file_read(path: str, max_bytes: int = 50_000) -> str:
    p = Path(path).expanduser()
    data = p.read_bytes()
    if len(data) > max_bytes:
        return data[:max_bytes].decode("utf-8", errors="replace") + f"\n... [truncated at {max_bytes} bytes]"
    return data.decode("utf-8", errors="replace")


def _file_write(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} bytes to {p}"


def _shell_exec(command: str, timeout: int = 30) -> str:
    """Run a shell command and return stdout+stderr.

    This is the most dangerous builtin. In a real deployment, restrict via
    container, allowlist, or a more granular tool surface (file_list, git_log,
    etc.) instead of giving the model a raw shell.
    """
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = result.stdout
    if result.stderr:
        out += f"\n[stderr]\n{result.stderr}"
    out += f"\n[exit {result.returncode}]"
    return out


def _calculator(expression: str) -> str:
    """Evaluate a math expression safely (no builtins, no names)."""
    # eval with empty globals/locals so only literals and operators work.
    # No `__import__`, no `open`, no anything. The model still can't break out.
    return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 — sandboxed


def _web_fetch(url: str, max_bytes: int = 20_000) -> str:
    """Fetch a URL and return its text content (truncated).

    Uses httpx if installed (in the `agents` extra), falls back to stdlib.
    No HTML parsing — the model can read raw HTML well enough for most tasks.
    """
    try:
        import httpx

        r = httpx.get(url, timeout=15, follow_redirects=True)
        text = r.text
    except ImportError:
        from urllib.request import Request, urlopen

        with urlopen(Request(url, headers={"User-Agent": "ai-playground-agent"}), timeout=15) as r:
            text = r.read().decode("utf-8", errors="replace")

    if len(text) > max_bytes:
        text = text[:max_bytes] + f"\n... [truncated at {max_bytes} chars]"
    return text


def builtin_tools(*, include_shell: bool = True, include_web: bool = True) -> list[Tool]:
    """Return the starter set of tools.

    Disable shell or web access if you're worried about blast radius — the
    agent works fine with just file + calculator for a lot of tasks.
    """
    tools = [
        Tool(
            name="file_read",
            description="Read a UTF-8 text file from disk and return its contents.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "max_bytes": {
                        "type": "integer",
                        "description": "Truncate after this many bytes.",
                        "default": 50_000,
                    },
                },
                "required": ["path"],
            },
            func=_file_read,
        ),
        Tool(
            name="file_write",
            description="Write text content to a file, creating parent directories if needed.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            func=_file_write,
        ),
        Tool(
            name="calculator",
            description="Evaluate a Python math expression (e.g. '2 ** 10 + sqrt(16)'). "
            "No imports, no function calls — just arithmetic operators and literals.",
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            func=_calculator,
        ),
    ]
    if include_shell:
        tools.append(
            Tool(
                name="shell_exec",
                description="Run a shell command and return stdout, stderr, and exit code.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer", "default": 30},
                    },
                    "required": ["command"],
                },
                func=_shell_exec,
            )
        )
    if include_web:
        tools.append(
            Tool(
                name="web_fetch",
                description="Fetch a URL and return its raw text/HTML body, truncated.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "max_bytes": {"type": "integer", "default": 20_000},
                    },
                    "required": ["url"],
                },
                func=_web_fetch,
            )
        )
    return tools


__all__ = ["Tool", "ToolRegistry", "ToolResult", "builtin_tools"]
