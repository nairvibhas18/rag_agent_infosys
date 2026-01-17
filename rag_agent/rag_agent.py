from dotenv import load_dotenv
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

load_dotenv()

model = OpenAIModel('gpt-4o')

agent = Agent(model=model)

result = agent.run_sync("What is the capital of France?")
print(result.data)







