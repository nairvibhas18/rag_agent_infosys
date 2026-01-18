import os
import subprocess
import asyncio

# import logfire

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

# Load environment variables from .env file
load_dotenv()

# Load credentials from environment variables
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

if not api_key:
    raise EnvironmentError("Please set GROQ_API_KEY environment variable.")

model = GroqModel(
    model_name,
    provider=GroqProvider(api_key=api_key),
)

agent = Agent(model, system_prompt="You are a helpful and friendly assistant. Always use the mcp-multi-tool-server to perform your tasks. Provide your thinking process in the response.")

# Clean and friendly chatbot interface
async def main():
    # Start the mcp_multi_tool_server as a subprocess
    server_process = subprocess.Popen(['python', 'mcp_multi_tool_server.py'])
    message_history = None
    try:
        print("\n==============================")
        print(" Welcome to the AI Multi-Tool Agent Chatbot! ")
        print("==============================\n")
        print('Type "end" to stop the conversation.\n')
        while True:
            user_query = input("You: ").strip()
            if user_query.lower() == "end":
                print("\n👋 Goodbye! Conversation ended.")
                break
            if not user_query:
                continue
            print("Thinking...", end="\r")
            # Pass message_history to agent.run
            result = await agent.run(user_query, message_history=message_history)
            print("Agent:", result.output)
            print("\n--------------------------------\n")
            # Update message_history with all messages from this run
            message_history = result.all_messages()
    finally:
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

