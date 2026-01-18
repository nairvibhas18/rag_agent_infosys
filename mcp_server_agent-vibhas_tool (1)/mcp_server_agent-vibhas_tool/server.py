import sqlite3
from mcp.server.fastmcp import FastMCP

DB_PATH = "chinook.db"  # Path to the Chinook SQLite database

server = FastMCP("Chinook Text-to-SQL Server")

@server.tool()
def execute_sql(sql: str) -> str:
    """Execute SQL SELECT queries on the Chinook database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(sql)
        rows = c.fetchall()
        columns = [desc[0] for desc in c.description] if c.description else []
        conn.close()
        
        if not rows:
            return "No results."
        
        # Format as table with column headers
        result = [" | ".join(columns)]
        result.append("-" * sum(len(col)+3 for col in columns))
        for row in rows:
            result.append(" | ".join(str(x).ljust(len(col)) for x, col in zip(row, columns)))
        return "\n".join(result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    server.run()
