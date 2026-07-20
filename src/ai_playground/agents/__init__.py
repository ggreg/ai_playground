"""Build-an-AI-agent-from-scratch module.

A from-scratch implementation of an LLM-driven agent. The goal is to make the
pieces of "agent" plain: a loop around a model that can call tools, remember
things, plan, and (optionally) cooperate with other agents.

The module deliberately mirrors the API patterns the major frameworks (LangChain,
LlamaIndex, Anthropic Managed Agents) converge on, but every piece is implemented
in less code than you'd find in any of them so you can read it end-to-end.

Layers, from low to high:

    llm.py         backend abstraction (Claude API, local transformer)
    tools.py       Tool dataclass + ToolRegistry + a few builtin tools
    memory.py      ConversationMemory (short term), VectorMemory (long term)
    agent.py       Agent class — the ReAct loop, the heart of the module
    planner.py     plan-and-execute pattern layered on top of Agent
    multi_agent.py supervisor + worker orchestration

See notebooks/06_agents/ for narrated walkthroughs.
"""

from .agent import Agent, AgentResult, AgentStep
from .llm import ClaudeBackend, LLMBackend, LLMResponse, LocalTransformerBackend
from .memory import ConversationMemory, Message, VectorMemory
from .tools import Tool, ToolRegistry, ToolResult, builtin_tools

__all__ = [
    "Agent",
    "AgentResult",
    "AgentStep",
    "ClaudeBackend",
    "ConversationMemory",
    "LLMBackend",
    "LLMResponse",
    "LocalTransformerBackend",
    "Message",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "VectorMemory",
    "builtin_tools",
]
