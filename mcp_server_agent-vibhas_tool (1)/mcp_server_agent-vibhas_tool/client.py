import os
import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncAzureOpenAI

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

model = OpenAIModel(
    deployment,  # Use the deployment name from environment variables
    provider=OpenAIProvider(openai_client=client),
)

# Connect to the local FastMCP server via stdio
server = MCPServerStdio(command="python", args=["server.py"])
agent = Agent(model, mcp_servers=[server])

CHINOOK_SCHEMA = """
Chinook Database Schema:
- albums (AlbumId, Title, ArtistId)
- artists (ArtistId, Name)
- customers (CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
- employees (EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
- genres (GenreId, Name)
- invoices (InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
- invoice_items (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
- media_types (MediaTypeId, Name)
- playlists (PlaylistId, Name)
- playlist_track (PlaylistId, TrackId)
- tracks (TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)

Key Relationships:
- albums.ArtistId → artists.ArtistId
- tracks.AlbumId → albums.AlbumId
- invoices.CustomerId → customers.CustomerId
- invoice_items.TrackId → tracks.TrackId
"""

SYSTEM_PROMPT = f"""You are an expert SQLite assistant for the Chinook database.\nSchema details:\n{CHINOOK_SCHEMA}\nConvert the user's request into a valid SQLite SELECT query.\nFollow these rules:\n1. Use table aliases for joins\n2. Prefer explicit JOIN syntax over implicit\n3. Only return valid SQL, no explanations\n4. Use LIMIT 10 unless specified otherwise"""

async def main():
    async with agent.run_mcp_servers():
        print("Chinook SQL Assistant ready. Ask about music sales! Type 'exit' to quit.")
        while True:
            user_input = input("\nQuery: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            try:
                # Ask the agent to generate SQL and execute it using the MCP tool
                prompt = f"Convert this to SQL: {user_input}"
                result = await agent.run(prompt, system=SYSTEM_PROMPT)
                print(f"\nGenerated SQL:\n{result.output}")
                # Call the tool to execute the SQL
                tool_result = await agent.call_tool("execute_sql", {"sql": result.output})
                print("\nResults:")
                print(tool_result.content.text)
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
