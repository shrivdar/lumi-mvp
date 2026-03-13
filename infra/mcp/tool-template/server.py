"""MCP tool server template.

Replace the example tool below with your own implementation.
See tool.json for the tool manifest.
"""

import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

TOOL_NAME = os.getenv("MCP_TOOL_NAME", "example-tool")
TOOL_PORT = int(os.getenv("MCP_TOOL_PORT", "8080"))

server = Server(TOOL_NAME)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name=TOOL_NAME,
            description="Example MCP tool — replace with your implementation",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Input query",
                    }
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != TOOL_NAME:
        raise ValueError(f"Unknown tool: {name}")

    query = arguments.get("query", "")
    # TODO: Implement your tool logic here
    return [TextContent(type="text", text=f"Result for: {query}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
