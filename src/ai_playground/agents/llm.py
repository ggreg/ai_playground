"""LLM backend abstraction — swap the brain without changing the agent.

The agent only needs one capability from the model: "given a chat history and a
list of tool schemas, return either text or a request to call a tool". Two
backends implement this:

- `ClaudeBackend` — calls the Anthropic API. Production-ready, handles
  multi-turn tool use natively (Claude returns `tool_use` blocks).
- `LocalTransformerBackend` — uses this repo's tiny LLaMA-style transformer.
  Educational: shows what the LLM *actually does* during an agent turn (it's
  just sampling tokens; tool calls are a JSON convention you parse out).

The split lets you keep the agent loop identical while switching the model. The
small transformer won't actually be good at tool use — that's part of the
lesson. Tool-following is an emergent capability of well-trained large models;
a 10M-parameter model trained on 100k tokens of toy text doesn't have it.

References:
- Anthropic tool use docs: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Toolformer (Schick et al., 2023): https://arxiv.org/abs/2302.04761
  — first demonstration that LMs can learn to call APIs via self-supervision
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..models.transformer import Transformer
    from .tools import Tool


@dataclass
class ToolCall:
    """A request from the model to invoke a tool.

    The `id` is echoed back when supplying the tool's result so the model can
    match call → result. Anthropic returns this id natively; for the local
    backend we generate one.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """One model turn. Either text, tool calls, or both.

    `stop_reason` mirrors Anthropic's field: "end_turn", "tool_use",
    "max_tokens", or "stop_sequence". The agent loop branches on this.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw: Any = None  # backend-specific response object, for debugging


class LLMBackend(Protocol):
    """The single interface the agent depends on.

    Keeping this surface minimal is deliberate: anything more (streaming,
    log probs, embeddings) would couple the agent to a specific provider.
    """

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Run one turn.

        `messages` follows the Anthropic format: list of
        `{"role": "user"|"assistant", "content": <str | list of blocks>}`.
        Using blocks (not flat strings) is required so tool_use/tool_result
        round-trips correctly.
        """
        ...


class ClaudeBackend:
    """Anthropic API backend.

    Uses the official `anthropic` SDK. Requires `ANTHROPIC_API_KEY` in env.
    Install via the `agents` extra: `uv sync --extra agents`.

    Default model is Sonnet 4.6 — a good balance of capability and cost for
    agent loops. Bump to Opus for harder reasoning, drop to Haiku for cheap
    high-volume tasks.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "ClaudeBackend requires the `anthropic` package. "
                "Install via: uv sync --extra agents"
            ) from e

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [t.to_anthropic_schema() for t in tools]

        response = self.client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            raw=response,
        )


class LocalTransformerBackend:
    """Uses the repo's tiny transformer as the agent's brain.

    Tool calls are signalled with a simple JSON convention the model is asked
    (in the system prompt) to emit:

        <tool_call>{"name": "...", "arguments": {...}}</tool_call>

    We parse those out post-hoc. A well-trained large model can be coaxed into
    following this format reliably; a tiny untrained model cannot — that's a
    feature for the learning notebook, not a bug.

    Why this matters: it makes concrete what an LLM "tool call" actually is.
    There's no special mechanism — the model emits text in a structured shape
    and a wrapper parses it. The Anthropic API hides this behind the
    `tool_use` block, but underneath it's a fine-tuned habit of emitting a
    particular token sequence.
    """

    def __init__(self, model: Transformer, tokenizer: Any, max_new_tokens: int = 256):
        self.model = model
        self.tokenizer = tokenizer  # any object with .encode(str) -> list[int] and .decode(list[int]) -> str
        self.max_new_tokens = max_new_tokens

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        import torch

        from ..inference.generate import generate

        prompt = self._format_prompt(messages, tools, system)
        tokens = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        device = next(self.model.parameters()).device
        tokens = tokens.to(device)

        out = generate(self.model, tokens, max_new_tokens=min(max_tokens, self.max_new_tokens))
        generated_ids = out[0, tokens.shape[1] :].tolist()
        text = self.tokenizer.decode(generated_ids)

        tool_calls = self._parse_tool_calls(text)
        # Strip the tool-call markers from the user-visible text
        clean_text = text
        for marker in ("<tool_call>", "</tool_call>"):
            clean_text = clean_text.replace(marker, "")

        return LLMResponse(
            text=clean_text.strip(),
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            raw=text,
        )

    @staticmethod
    def _format_prompt(
        messages: list[dict[str, Any]],
        tools: list[Tool] | None,
        system: str | None,
    ) -> str:
        """Flatten the chat into a single string with a simple chat template.

        Real models use trained chat templates (Llama-3 uses `<|start_header|>`
        etc.); ours is a toy text format. The point is the *shape*: system
        prompt, then alternating user/assistant turns, then an `assistant:`
        priming cue.
        """
        parts: list[str] = []
        if system:
            parts.append(f"system: {system}")
        if tools:
            schemas = [t.to_anthropic_schema() for t in tools]
            parts.append("Available tools (emit <tool_call>{json}</tool_call> to call):")
            parts.append(json.dumps(schemas, indent=2))
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                # Flatten content blocks (text only — local model can't see images)
                content = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
            parts.append(f"{msg['role']}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    @staticmethod
    def _parse_tool_calls(text: str) -> list[ToolCall]:
        import re
        import uuid

        calls: list[ToolCall] = []
        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
            try:
                payload = json.loads(match.group(1))
                calls.append(
                    ToolCall(
                        id=str(uuid.uuid4())[:8],
                        name=payload["name"],
                        arguments=payload.get("arguments", {}),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue  # ignore malformed calls — common with small models
        return calls
