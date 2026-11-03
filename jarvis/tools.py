"""Tool registry for Jarvis assistant."""

import logging
from typing import Dict, Callable, Any
import json

_logger = logging.getLogger("jarvis.tools")


class ToolRegistry:
    """A registry for tools that can be called by the LLM."""
    
    def __init__(self):
        self._tools: Dict[str, Dict] = {}
        
    def register(self, name: str, func: Callable, schema: Dict[str, Any]) -> None:
        """Register a tool.
        
        Args:
            name: The name of the tool
            func: The function to call when this tool is invoked
            schema: The JSON schema for the tool (used for OpenAI tool calling)
        """
        self._tools[name] = {
            "function": func,
            "schema": schema
        }
        _logger.debug(f"Registered tool: {name}")
        
    def get_tool(self, name: str) -> Dict[str, Any]:
        """Get a tool by name.
        
        Args:
            name: The name of the tool to retrieve
            
        Returns:
            Dictionary with function and schema for the tool
        """
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found")
        return self._tools[name]
        
    def get_all_tools(self) -> Dict[str, Any]:
        """Get all registered tools.
        
        Returns:
            Dictionary of all tools with their schemas
        """
        # Return just the schemas for OpenAI tool calling format
        return {name: tool["schema"] for name, tool in self._tools.items()}
        
    def get_tool_names(self) -> list:
        """Get a list of all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())


# Global tool registry
registry = ToolRegistry()


def tool(name: str, schema: Dict[str, Any]):
    """Decorator to register a function as a tool.
    
    Args:
        name: The name of the tool (used for invocation)
        schema: The JSON schema for the tool parameters
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Set up the default schema if not provided
        if not schema.get("name"):
            schema["name"] = name
            
        registry.register(name, func, schema)
        return func
    return decorator


# Register our dummy tool - this is for testing and demo purposes only
@tool(
    name="get_demo_number",
    schema={
        "name": "get_demo_number",
        "description": "Get a demo number for a given name (for testing purposes)",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name to look up"
                }
            },
            "required": ["name"]
        }
    }
)
def get_demo_number(name: str) -> int:
    """Get a demo number for a given name.
    
    For demo purposes only - returns different numbers based on the name.
    
    Args:
        name: The name to look up
        
    Returns:
        A fixed number associated with the name
    """
    # Simple mapping for testing
    name_mapping = {
        "asha": 42,
        "john": 13,
        "mary": 99,
        "alex": 7,
        "sarah": 28
    }
    
    return name_mapping.get(name.lower(), 42)  # Default to 42 if name not found