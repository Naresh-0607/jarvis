"""Tests for tool functionality."""

import pytest
from jarvis.tools import registry, tool


def test_tool_registration():
    """Test that tools can be registered."""
    
    @tool(
        name="test_tool", 
        schema={
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_param": {"type": "string"}
                },
                "required": ["test_param"]
            }
        }
    )
    def test_function(test_param: str) -> str:
        return f"Processed: {test_param}"
        
    # Should have registered the tool
    assert "test_tool" in registry.get_tool_names()
    
    # Should be able to get the tool back
    tool_info = registry.get_tool("test_tool")
    assert "function" in tool_info
    assert "schema" in tool_info
    
    # Test that we can call it
    result = test_function("hello")
    assert result == "Processed: hello"


def test_tool_schema():
    """Test tool schema format."""
    
    @tool(
        name="schema_test", 
        schema={
            "name": "schema_test",
            "description": "Test tool with schema",
            "parameters": {
                "type": "object", 
                "properties": {
                    "required_field": {"type": "string"},
                    "optional_field": {"type": "number"}
                },
                "required": ["required_field"]
            }
        }
    )
    def schema_function(required_field: str, optional_field: int = 0) -> str:
        return f"Result: {required_field}"
        
    # Should have the correct tool names
    tool_names = registry.get_tool_names()
    assert "schema_test" in tool_names
    
    # Test tool availability
    all_tools = registry.get_all_tools()
    assert "schema_test" in all_tools


def test_extract_raw_tool_calls_fallback():
    """Verify that text-based tool calls emitted by Ollama/Qwen models are parsed."""
    from jarvis.chat import ChatSession

    session = ChatSession()
    sample_text = (
        "Let me check your reminders for you.\n\n"
        "<function=list_tasks>\n"
        "<parameter=range>\n"
        "today\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    calls = session._extract_raw_tool_calls(sample_text)
    assert len(calls) == 1
    assert calls[0].name == "list_tasks"
    assert '"range": "today"' in calls[0].arguments

    clean = session._strip_raw_tool_calls(sample_text)
    assert clean == "Let me check your reminders for you."
    assert "<function" not in clean
    assert "</tool_call>" not in clean