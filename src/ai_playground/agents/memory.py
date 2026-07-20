"""Memory — how the agent remembers things across turns and conversations.

Two complementary kinds:

1. **Short-term / working memory** (`ConversationMemory`): the chat history
   itself. Constrained by context window. When it grows too large we
   *summarize* older turns instead of dropping them.

2. **Long-term memory** (`VectorMemory`): a separate store, retrieved by
   semantic similarity. Things move here either deliberately ("remember that
   X") or via end-of-session summarization. The agent queries it when a
   topic surfaces again later.

These mirror how production agents do it (ChatGPT's "memories" feature,
LangChain's ConversationSummaryBufferMemory, Claude's Memory tool):

    | layer           | scope          | lookup       | typical size  |
    | --------------- | -------------- | ------------ | ------------- |
    | working memory  | this session   | sequential   | <100k tokens  |
    | long-term store | all sessions   | similarity   | unbounded     |

References:
- MemGPT (Packer et al., 2023): https://arxiv.org/abs/2310.08560
  — formalises the two-tier idea with explicit "function calls" to swap
  pages of memory in and out of the context window.
- Generative Agents (Park et al., 2023): https://arxiv.org/abs/2304.03442
  — memory stream + reflection, the canonical research demo of long-term
  agent memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Message:
    """A single turn in the conversation.

    `content` is either a string (simple turn) or a list of content blocks
    (tool_use / tool_result turns). Matches the Anthropic message format so
    we can pass it straight through to the API.
    """

    role: str  # "user" | "assistant" | "system"
    content: str | list[dict[str, Any]]
    timestamp: float = field(default_factory=time.time)

    def to_api(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """Bounded chat history with optional summarization of evicted turns.

    Strategy: keep the most recent `keep_turns` turns verbatim. When a turn
    falls off the end, optionally summarize it into a running summary that
    sits at the front of the message list as a system note.

    For real production agents you'd want token-aware truncation (count actual
    tokens, drop when over budget). Turn-count is simpler and good enough for
    learning. The pattern is the same either way.
    """

    def __init__(self, keep_turns: int = 20, summarizer: Any = None):
        self.keep_turns = keep_turns
        self.summarizer = summarizer  # an LLMBackend used to compress evicted turns; optional
        self._messages: deque[Message] = deque()
        self._summary: str = ""

    def add(self, role: str, content: str | list[dict[str, Any]]) -> None:
        self._messages.append(Message(role=role, content=content))
        while len(self._messages) > self.keep_turns:
            evicted = self._messages.popleft()
            if self.summarizer is not None:
                self._summary = self._extend_summary(evicted)

    def messages(self) -> list[dict[str, Any]]:
        """Return the message list in API format, with summary prepended."""
        out: list[dict[str, Any]] = []
        if self._summary:
            out.append({
                "role": "user",
                "content": f"[earlier conversation summary]\n{self._summary}",
            })
        out.extend(m.to_api() for m in self._messages)
        return out

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""

    def _extend_summary(self, evicted: Message) -> str:
        """Roll the evicted message into the running summary.

        This is a one-shot LLM call per eviction — fine for a learning toy,
        wasteful for production. Real systems batch evictions and summarize
        on a timer or when context is about to overflow.
        """
        content_str = (
            evicted.content
            if isinstance(evicted.content, str)
            else json.dumps(evicted.content)
        )
        prompt = (
            f"Existing summary:\n{self._summary or '(none)'}\n\n"
            f"New turn to incorporate ({evicted.role}):\n{content_str}\n\n"
            "Rewrite the summary to incorporate the new turn. "
            "Keep it under 200 words. Preserve concrete facts, decisions, and open questions."
        )
        response = self.summarizer.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return response.text.strip()


# ---------------------------------------------------------------------------
# Long-term memory: a tiny vector store
# ---------------------------------------------------------------------------


def _hash_embed(text: str, dim: int = 256) -> list[float]:
    """A toy embedding: hashing trick over whitespace tokens.

    This is *not* a real semantic embedding. It captures lexical overlap
    (same words → similar vectors) but not meaning (synonyms → unrelated).
    Used as the default so the memory module works offline with no extra
    dependencies — for production-quality recall you'd wire in real
    embeddings (Cohere, OpenAI, or a local sentence-transformer).

    The hashing trick: every token gets a deterministic hash, the hash picks
    a dimension to bump. Word frequency becomes a sparse-ish vector. L2-
    normalize so cosine similarity == dot product.
    """
    vec = [0.0] * dim
    tokens = re.findall(r"\w+", text.lower())
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass
class MemoryEntry:
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class VectorMemory:
    """A tiny in-memory vector store.

    Two operations: `add(text)` and `search(query, top_k)`. Persistable as
    JSON so you can save and reload memories between sessions. Brute-force
    search is fine up to ~10k entries; past that you'd want FAISS or hnswlib.

    Embedding function is pluggable — swap in a real model when you have one:

        VectorMemory(embed_fn=lambda t: anthropic.embed(t))
    """

    def __init__(self, embed_fn: Any = None, path: str | Path | None = None):
        self.embed_fn = embed_fn or _hash_embed
        self.path = Path(path) if path else None
        self._entries: list[MemoryEntry] = []
        if self.path and self.path.exists():
            self._load()

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        entry = MemoryEntry(
            text=text,
            embedding=self.embed_fn(text),
            metadata=metadata or {},
        )
        self._entries.append(entry)
        if self.path:
            self._save()

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, MemoryEntry]]:
        if not self._entries:
            return []
        q_emb = self.embed_fn(query)
        scored = [(_cosine(q_emb, e.embedding), e) for e in self._entries]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._entries)

    def _save(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            [
                {
                    "text": e.text,
                    "embedding": e.embedding,
                    "metadata": e.metadata,
                    "timestamp": e.timestamp,
                }
                for e in self._entries
            ]
        ))

    def _load(self) -> None:
        assert self.path is not None
        data = json.loads(self.path.read_text())
        self._entries = [
            MemoryEntry(
                text=d["text"],
                embedding=d["embedding"],
                metadata=d.get("metadata", {}),
                timestamp=d.get("timestamp", time.time()),
            )
            for d in data
        ]


__all__ = ["ConversationMemory", "Message", "MemoryEntry", "VectorMemory"]
