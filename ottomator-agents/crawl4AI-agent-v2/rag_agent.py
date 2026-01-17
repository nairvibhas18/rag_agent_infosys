"""Pydantic AI agent that leverages RAG with a local ChromaDB for Pydantic documentation."""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional
import asyncio
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
#from pydantic_ai.models.groq import GroqModel

from utils import (
    get_chroma_client,
    get_or_create_collection,
    query_collection,
    format_results_as_context,
    ingest_pdf_to_chromadb,
)

# Load environment variables from .env file
dotenv.load_dotenv()

# Check for Groq API key
if not os.getenv("GROQ_API_KEY"):
    print("Error: GROQ_API_KEY environment variable not set.")
    print("Please create a .env file with your Groq API key or set it in your environment.")
    sys.exit(1)


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    chroma_client: chromadb.PersistentClient
    collection_name: str
    embedding_model: str
    embedding_func: callable


# Create the RAG agent
agent = Agent(
    'groq:llama-3.3-70b-versatile',
    deps_type=RAGDeps,
    system_prompt="You are a helpful assistant that answers questions based on the provided CONTEXT INFORMATION. Always answer using the provided context above. Do not call tools or output tool call syntax. If the context does not contain the answer, clearly state that the information isn't available in the current documentation and provide your best general knowledge response."
)


@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str, n_results: int = 5) -> str:
    """Retrieve relevant documents from ChromaDB based on a search query.
    
    Args:
        context: The run context containing dependencies.
        search_query: The search query to find relevant documents.
        n_results: Number of results to return (default: 5).
        
    Returns:
        Formatted context information from the retrieved documents.
    """
    # Get ChromaDB client and collection
    collection = get_or_create_collection(
        context.deps.chroma_client,
        context.deps.collection_name,
        embedding_model_name=context.deps.embedding_model
    )
    
    # Query the collection
    query_results = query_collection(
        collection,
        search_query,
        n_results=n_results
    )
    
    # Format the results as context
    context = format_results_as_context(query_results)
    return context


@agent.tool
async def add_pdf(context: RunContext, url: str) -> str:
    """Ingest a PDF from a URL into the ChromaDB collection."""
    # Get ChromaDB collection and embedding function from context.deps
    collection = get_or_create_collection(
        context.deps.chroma_client,
        context.deps.collection_name,
        embedding_model_name=context.deps.embedding_model
    )
    embedding_func = context.deps.embedding_func
    await ingest_pdf_to_chromadb(url, collection, embedding_func)
    return "PDF ingested successfully."


async def run_rag_agent(
    question: str,
    collection_name: str = "docs",
    db_directory: str = "./chroma_db",
    embedding_model: str = "all-MiniLM-L6-v2",
    n_results: int = 5,
    return_context: bool = False
) -> str:
    """Run the RAG agent to answer a question about Pydantic AI.
    
    Args:
        question: The question to answer.
        collection_name: Name of the ChromaDB collection to use.
        db_directory: Directory where ChromaDB data is stored.
        embedding_model: Name of the embedding model to use.
        n_results: Number of results to return from the retrieval.
        return_context: If True, return a tuple (answer, context)
        
    Returns:
        The agent's response, or (response, context) if return_context is True.
    """
    # Create dependencies
    deps = RAGDeps(
        chroma_client=get_chroma_client(db_directory),
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_func=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    )
    # Manually perform retrieval using the same logic as the tool
    collection = get_or_create_collection(
        deps.chroma_client,
        deps.collection_name,
        embedding_model_name=deps.embedding_model
    )
    # Try to get PDF chunks first
    query_results = query_collection(
        collection,
        question,
        n_results=n_results,
        where={"source": "uploaded"}
    )
    # If not enough results, fall back to all docs
    if not query_results["documents"][0]:
        query_results = query_collection(
            collection,
            question,
            n_results=n_results
        )
    context = format_results_as_context(query_results)
    # 2. Prepend context to the question
    prompt = f"{context}\n\nUser question: {question}"
    # 3. Run the agent with the context-augmented prompt
    result = await agent.run(prompt, deps=deps)
    if return_context:
        return result.data, context
    return result.data


def main():
    """Main function to parse arguments and run the RAG agent."""
    parser = argparse.ArgumentParser(description="Run a Pydantic AI agent with RAG using ChromaDB")
    parser.add_argument("--question", help="The question to answer about Pydantic AI")
    parser.add_argument("--collection", default="docs", help="Name of the ChromaDB collection")
    parser.add_argument("--db-dir", default="./chroma_db", help="Directory where ChromaDB data is stored")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Name of the embedding model to use")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return from the retrieval")
    
    args = parser.parse_args()
    
    # Run the agent
    response = asyncio.run(run_rag_agent(
        args.question,
        collection_name=args.collection,
        db_directory=args.db_dir,
        embedding_model=args.embedding_model,
        n_results=args.n_results
    ))
    
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
