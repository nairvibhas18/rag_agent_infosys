# text_to_sql_code_tool.py
"""
A tool for converting natural language text to SQL queries and executing them on a SQLite database, with structured output and validation.
"""

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Union
from annotated_types import MinLen
from pydantic import BaseModel, Field
from typing_extensions import TypeAlias
from pydantic_ai import Agent, ModelRetry, RunContext, format_as_xml
import sqlite3
import os
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Example SQLite schema and examples
DB_SCHEMA = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT
);
"""
SQL_EXAMPLES = [
    {
        'request': 'show me all users',
        'response': "SELECT * FROM users;",
    },
    {
        'request': 'show me users with email ending in gmail.com',
        'response': "SELECT * FROM users WHERE email LIKE '%gmail.com';",
    },
]

class Success(BaseModel):
    sql_query: Annotated[str, MinLen(1)]
    explanation: str = Field('', description='Explanation of the SQL query, as markdown')
    result: str = Field('', description='Result of the SQL query, as natural language')

class InvalidRequest(BaseModel):
    error_message: str

Response: TypeAlias = Union[Success, InvalidRequest]

@dataclass
class Deps:
    conn: sqlite3.Connection

load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
if not all([api_key, endpoint, deployment]):
    raise EnvironmentError("Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT environment variables.")
client = AsyncAzureOpenAI(
    azure_endpoint=endpoint,
    api_version=api_version,
    api_key=api_key,
)

agent: Agent[Deps, Response] = Agent(
    OpenAIModel(
        deployment,
        provider=OpenAIProvider(openai_client=client),
    ),
    output_type=Response,  # type: ignore
    deps_type=Deps,
)

@agent.system_prompt
def system_prompt() -> str:
    return f"""
Given the following SQLite table of users, your job is to write a SQL query that suits the user's request.

Database schema:

{DB_SCHEMA}

today's date = {date.today()}

{format_as_xml(SQL_EXAMPLES)}
"""

@agent.output_validator
def validate_output(ctx: RunContext[Deps], output: Response) -> Response:
    if isinstance(output, InvalidRequest):
        return output
    output.sql_query = output.sql_query.replace('\\', '')
    if not output.sql_query.upper().startswith('SELECT'):
        raise ModelRetry('Please create a SELECT query')
    try:
        ctx.deps.conn.execute(f'EXPLAIN {output.sql_query}')
    except Exception as e:
        raise ModelRetry(f'Invalid query: {e}') from e
    else:
        return output

async def text_to_sql_code(query: str) -> Response:
    """Convert a natural language query to an SQL statement, execute it on SQLite, and return a structured response."""
    conn = sqlite3.connect('chinook.db')
    deps = Deps(conn)
    try:
        result = await agent.run(query, deps=deps)
        # Actually run the generated SQL if valid
        if isinstance(result.output, Success):
            try:
                cursor = conn.cursor()
                cursor.execute(result.output.sql_query)
                rows = cursor.fetchall()
                if not rows:
                    result.output.result = "No results found."
                else:
                    result.output.result = "Here are the results: " + ", ".join(str(row) for row in rows)
            except Exception as e:
                return InvalidRequest(error_message=f"Database error: {e}")
            return result.output
        else:
            return result.output
    finally:
        conn.close()
