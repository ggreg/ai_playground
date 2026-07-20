"""Tests for the tool abstraction and builtin tools."""

import pytest

from ai_playground.agents.tools import Tool, ToolRegistry, builtin_tools


class TestToolRegistry:
    def test_register_and_get(self):
        tool = Tool(
            name="echo",
            description="echo back",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            func=lambda x: x,
        )
        reg = ToolRegistry([tool])
        assert reg.get("echo") is tool
        assert reg.get("missing") is None

    def test_double_register_raises(self):
        tool = Tool(name="t", description="", input_schema={}, func=lambda: None)
        reg = ToolRegistry([tool])
        with pytest.raises(ValueError):
            reg.register(tool)

    def test_dispatch_success(self):
        tool = Tool(
            name="add",
            description="add",
            input_schema={},
            func=lambda a, b: a + b,
        )
        reg = ToolRegistry([tool])
        result = reg.dispatch("add", {"a": 2, "b": 3}, call_id="c1")
        assert result.content == "5"
        assert result.is_error is False
        assert result.tool_call_id == "c1"

    def test_dispatch_unknown_tool(self):
        reg = ToolRegistry()
        result = reg.dispatch("nope", {}, call_id="c1")
        assert result.is_error is True
        assert "unknown tool" in result.content

    def test_dispatch_exception_captured(self):
        tool = Tool(name="bad", description="", input_schema={}, func=lambda: 1 / 0)
        reg = ToolRegistry([tool])
        result = reg.dispatch("bad", {}, call_id="c1")
        assert result.is_error is True
        assert "ZeroDivisionError" in result.content


class TestBuiltinTools:
    def test_file_read_write_roundtrip(self, tmp_path):
        tools = ToolRegistry(builtin_tools(include_shell=False, include_web=False))
        target = tmp_path / "x.txt"

        write_result = tools.dispatch(
            "file_write", {"path": str(target), "content": "hello"}, call_id="w"
        )
        assert write_result.is_error is False
        assert target.read_text() == "hello"

        read_result = tools.dispatch("file_read", {"path": str(target)}, call_id="r")
        assert read_result.content == "hello"

    def test_calculator_basic(self):
        tools = ToolRegistry(builtin_tools(include_shell=False, include_web=False))
        result = tools.dispatch("calculator", {"expression": "2 ** 10 + 24"}, call_id="c")
        assert result.content == "1048"

    def test_calculator_blocks_dunder(self):
        """The sandbox should prevent access to dangerous builtins."""
        tools = ToolRegistry(builtin_tools(include_shell=False, include_web=False))
        result = tools.dispatch(
            "calculator",
            {"expression": "__import__('os').system('echo pwned')"},
            call_id="c",
        )
        assert result.is_error is True

    def test_schema_shape(self):
        tools = builtin_tools(include_shell=False, include_web=False)
        for t in tools:
            schema = t.to_anthropic_schema()
            assert set(schema.keys()) == {"name", "description", "input_schema"}
            assert schema["input_schema"]["type"] == "object"
