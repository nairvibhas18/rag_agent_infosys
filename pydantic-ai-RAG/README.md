# Create a RAG AI Agent witj PydancticAI, OpenAI and PostgreSQL

In this tutorial we will pick up where we left off. We built an [AI Agent that could perform CRUD operations](https://github.com/skolo-online/ai-agents) on a PostgreSQL DB. 

We will not create another agent that can do RAG (Retrieval-Augmented Generation) - to answer medical, health related questions using only the information we have given it. We will build a health consultatant using PydanticAI.

## Pre-Requisites

To get started you will need: 
* A PostgreSQL Database instance, you can use the one from the previous tutorial or create a new [one here on Digital Ocean](https://m.do.co/c/7d9a2c75356d ), then get the connection string to use in the code.
* You will also need an [OpenAI API Key](https://platform.openai.com/)
* IDE - Integrated Development Environment - I use cursor, you can [download for free here](https://www.cursor.com/)
* Python 3.9+ to work with PydanticAI
* An online PDF document, of if you have the document on yoru computer you can get online storage eg. Spaces Object Storage from digital ocean and upload your document there. Alternatively you can use this document we have already uploaded here: [https://skolo-ai-agent.ams3.cdn.digitaloceanspaces.com/pydantic/the_seven_realms.pdf](https://skolo-ai-agent.ams3.cdn.digitaloceanspaces.com/pydantic/the_seven_realms.pdf)

## Get Started

First of all, let us just confirm that we have python installed. Run this command"
```sh
python3 --version
```

And you should see your python3 version, the number should be 9 or above. In my case, I have 3.10.12
`Python 3.10.12`

Get your virtual environemnt started:

```sh
virtualenv skoloenv

source skoloenv/bin/activate
```

The install the pre-requisites

```sh
pip install pydantic-ai asyncpg httpx pymupdf
```

**Get the DB connection string**
""

**Get the OpenAI API Key**  
""

## The Code Details 

Create a file called `aiagent.py`

Copy this code below and replace the `DB_DSN` and the `OPENAI_API_KEY` 

You will need the following imports:

```python
from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List
import pydantic_core

import asyncpg
import httpx
import fitz 
import json
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from openai import AsyncOpenAI
```

The code for adding documents to the database using URL
Note: This will only work for PDF documents, adjust accordingly for other document types


```python
# Connection string for PostgreSQL database
DB_DSN = "database-dsn-goes-here"
OPENAI_API_KEY = "sk-proj-your-api-key-goes-here"
DB_SCHEMA = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS text_chunks (
        id serial PRIMARY KEY,
        chunk text NOT NULL,
        embedding vector(1536) NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_text_chunks_embedding ON text_chunks USING hnsw (embedding vector_l2_ops);
    """



@asynccontextmanager
async def database_connect(create_db: bool = False):
    """Manage database connection pool."""
    pool = await asyncpg.create_pool(DB_DSN)
    try:
        if create_db:
            async with pool.acquire() as conn:
                await conn.execute(DB_SCHEMA)
        yield pool
    finally:
        await pool.close()

class Chunk(BaseModel):
    chunk: str

async def split_text_into_chunks(text: str, max_words: int = 400, overlap: float = 0.2) -> List[Chunk]:
    """Split long text into smaller chunks based on word count with overlap."""
    words = text.split()
    chunks = []
    step_size = int(max_words * (1 - overlap))

    for start in range(0, len(words), step_size):
        end = start + max_words
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(Chunk(chunk=" ".join(chunk_words)))

    return chunks

async def insert_chunks(pool: asyncpg.Pool, chunks: List[Chunk], openai_client: AsyncOpenAI):
    """Insert text chunks into the database with embeddings."""
    for chunk in chunks:
        embedding_response = await openai_client.embeddings.create(
            input=chunk.chunk,
            model="text-embedding-3-small"
        )
        
        # Extract embedding data and convert to JSON format
        assert len(embedding_response.data) == 1, f"Expected 1 embedding, got {len(embedding_response.data)}"
        embedding_data = json.dumps(embedding_response.data[0].embedding)

        # Insert into the database
        await pool.execute(
            'INSERT INTO text_chunks (chunk, embedding) VALUES ($1, $2)',
            chunk.chunk,
            embedding_data 
        )

async def download_pdf(url: str) -> bytes:
    """Download PDF from a given URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF content."""
    document = fitz.open(stream=pdf_content, filetype="pdf")
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

async def add_pdf_to_db(url: str):
    """Download PDF, extract text, and add to the embeddings database."""
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    pdf_content = await download_pdf(url)
    text = extract_text_from_pdf(pdf_content)
    async with database_connect(create_db=True) as pool:
        chunks = await split_text_into_chunks(text)
        await insert_chunks(pool, chunks, openai_client)

async def update_db_with_pdf(url: str):
    """Download PDF, extract text, and update the embeddings database."""
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    pdf_content = await download_pdf(url)
    text = extract_text_from_pdf(pdf_content)
    async with database_connect() as pool:
        chunks = await split_text_into_chunks(text)
        await insert_chunks(pool, chunks, openai_client)


async def execute_url_pdf(url: str):
    """
    Check if the database table exists, and call the appropriate function
    to handle the PDF URL.
    """
    async with database_connect() as pool:
        table_exists = await pool.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'text_chunks'
            )
        """)

    if table_exists:
        # If the table exists, update the database
        print("Table exists. Updating database with PDF content.")
        await update_db_with_pdf(url)
    else:
        # If the table does not exist, add the PDF and create the table
        print("Table does not exist. Adding PDF and creating the table.")
        await add_pdf_to_db(url)

```

Then finally the AGENT code:

```python
@dataclass
class Deps:
    pool: asyncpg.Pool
    openai: AsyncOpenAI


# Initialize the agent
model = OpenAIModel("gpt-4o", api_key=OPENAI_API_KEY)
rag_agent = Agent(model, deps_type=Deps)
    

@rag_agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    print("Retrieving..............")
    embedding = await context.deps.openai.embeddings.create(
            input=search_query,
            model='text-embedding-3-small',
        )
    
    assert (
        len(embedding.data) == 1
    ), f'Expected 1 embedding, got {len(embedding.data)}, doc query: {search_query!r}'
    
    embedding = embedding.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding).decode()
    rows = await context.deps.pool.fetch(
        'SELECT chunk FROM text_chunks ORDER BY embedding <-> $1 LIMIT 5',
        embedding_json,
    )
    from_db = '\n\n'.join(
    f'# Chunk:\n{row["chunk"]}\n'
    for row in rows
    ) 
    return from_db

async def run_agent(question: str):
    """Entry point to run the agent and perform RAG-based question answering."""

    # Set up the agent and dependencies
    async with database_connect() as pool:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        async with database_connect(False) as pool:
            deps = Deps(openai=openai_client, pool=pool)
            base_instruction = f"Use the 'retrieve' tool to fetch information to help you answer this question: {question}"
            answer = await rag_agent.run(base_instruction, deps=deps)
            return answer.data
```

## Streamlit Application: Front-end

To interact with our agent, we will use the streamlit application below: 

Install Streamlit: 
```sh
pip install streamlit
```


Create a file called `app.py`

Add this code to `app.py`
```python
import streamlit as st
import asyncio
from aiagent import execute_url_pdf, run_agent

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Assistant 📚🤖",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("AI Assistant 📚🤖")
st.write("Interact with your PDF-based AI assistant. Use the options below to upload a PDF or ask a question.")

# Layout with Two Columns
col1, col2 = st.columns(2)

# Column 1: PDF Upload via URL
with col1:
    st.subheader("📄 Upload PDF")
    pdf_url = st.text_input("Enter the URL of the PDF document:", placeholder="https://example.com/document.pdf")
    if st.button("📥 Add PDF to Database"):
        if pdf_url:
            with st.spinner("Processing the PDF and updating the database..."):
                try:
                    asyncio.run(execute_url_pdf(pdf_url))
                    st.success("PDF successfully processed and added to the database!")
                except Exception as e:
                    st.error(f"Error processing the PDF: {e}")
        else:
            st.warning("Please enter a valid URL.")

# Column 2: Ask a Question
with col2:
    st.subheader("❓ Ask a Question")
    question = st.text_input("Enter your question:", placeholder="What are the responsibilities of a full-stack developer?")
    if st.button("🔍 Get Answer"):
        if question:
            with st.spinner("Thinking..."):
                try:
                    answer = asyncio.run(run_agent(question))
                    st.success("Here's the answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error getting the answer: {e}")
        else:
            st.warning("Please enter a valid question.")

# Footer
st.markdown("---")
st.write("✨ Powered by [Skolo Online](https://skolo.online) and Pydantic AI")

```

To run the application
```
streamlit run app.py
```





