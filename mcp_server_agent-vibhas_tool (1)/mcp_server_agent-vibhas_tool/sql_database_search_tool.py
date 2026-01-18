# sql_database_search_tool.py
"""
A tool for searching a SQLite database using a natural language query and returning the answer in natural language.
"""

import sqlite3

async def sql_database_search(query: str) -> str:
    """Search the SQLite database using a natural language query and return the answer in natural language."""
    print(f"Searching database for: {query}")
    # Very basic mapping for demonstration
    if "how many users" in query.lower():
        sql = "SELECT COUNT(*) FROM users;"
        answer_prefix = "There are"
    elif "list all users" in query.lower():
        sql = "SELECT name FROM users;"
        answer_prefix = "The users are:"
    else:
        return f"Sorry, I can't answer that query yet."

    try:
        conn = sqlite3.connect('chinook.db')
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        if not rows:
            return "No results found."
        if "count" in sql.lower():
            return f"{answer_prefix} {rows[0][0]} users."
        else:
            result = answer_prefix + " " + ", ".join(str(row[0]) for row in rows)
            return result
    except Exception as e:
        return f"Database error: {e}"
    finally:
        if 'conn' in locals():
            conn.close()
