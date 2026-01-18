# AzureChatOpenAI with pydanticai agent utilizing tools from MCP Server.

This project demonstrates how to use the `pydanticai` framework to call AzureChatOpenAI from Python. We have a set a tools on the MCP Server that the agent can use.

## Setup

1. Install dependencies:
#### Unfortunately, we are unable to use Docker or any sort of containerized development, so for now, we will use a virtual environment and manually install all dependencies. 
- First, enter the virtual environment:
   ```powershell
   python -m venv .venv
   ./.venv/Scripts/activate
   ```
- Next, install all required dependencies:
   ```powershell
   python -m pip install -r .\requirements.txt
   ```
- Create your own branch if you want to produce work and try not to push to main without reviewing code.

2. Set your Azure OpenAI credentials as environment variables:
   - `AZURE_OPENAI_API_KEY`
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_DEPLOYMENT`

## Usage

Run the script:
```powershell
python main.py
```

The script sends a prompt to AzureChatOpenAI and prints the response.
