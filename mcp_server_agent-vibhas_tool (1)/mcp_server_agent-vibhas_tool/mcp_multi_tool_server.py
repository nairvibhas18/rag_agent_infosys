from mcp.server.fastmcp import FastMCP
import sqlite3
from typing import List, Dict, Any

# Import the text_to_sql function from the text_to_sql_tool module
from text_to_sql_code_tool import text_to_sql_code
from sql_database_search_tool import sql_database_search

# Create the multi-tool server
mcp_multi_tool_server = FastMCP('mcp-multi-tool-server')

# addition calculator tool
@mcp_multi_tool_server.tool()
async def add_tool(a: float, b: float) -> float:
    """Add two numbers together."""
    print(f"Adding {a} and {b}")
    return a + b

# multiplication calculator tool
@mcp_multi_tool_server.tool()
async def multiply_tool(a: float, b: float) -> float:
    """Multiply two numbers together."""
    print(f"Multiplying {a} and {b}")
    return a * b

# text to SQL natural language conversion tool
@mcp_multi_tool_server.tool()
async def text_to_sql_tool(query: str) -> str:
    """Convert a natural language query to an SQL statement."""
    return await text_to_sql_code(query)

# SQL database search tool
@mcp_multi_tool_server.tool()
async def sql_database_search_tool(query: str) -> str:
    """Search the SQLite database using a natural language query and return the answer in natural language."""
    return await sql_database_search(query)


# must run this in order to start the server.
if __name__ == "__main__":
    print("Starting multi-tool MCP server...")
    mcp_multi_tool_server.run()