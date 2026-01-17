from dotenv import load_dotenv
import streamlit as st
import asyncio
import os
import nest_asyncio

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

from rag_agent import agent, RAGDeps, run_rag_agent
from utils import get_chroma_client, ingest_pdf_file_to_chromadb
from chromadb.utils import embedding_functions

load_dotenv()
nest_asyncio.apply()

def get_agent_deps():
    db_directory = "./chroma_db"
    collection_name = "docs"
    embedding_model = "all-MiniLM-L6-v2"
    from rag_agent import RAGDeps
    from utils import get_chroma_client
    chroma_client = get_chroma_client(db_directory)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    return RAGDeps(
        chroma_client=chroma_client,
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_func=embedding_func
    )


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # user-prompt
    if part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)             

async def run_agent_with_streaming(user_input):
    async with agent.run_stream(
        user_input, deps=st.session_state.agent_deps, message_history=st.session_state.messages
    ) as result:
        async for message in result.stream_text(delta=True):  
            yield message

    # Add the new messages to the chat history (including tool calls and responses)
    st.session_state.messages.extend(result.new_messages())


async def get_answer(question):
    return await run_rag_agent(question)

def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():
    st.title("AI Assistant 📚🤖")
    st.write("Interact with your AI assistant. Use the options below to upload a PDF or ask a question.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_deps" not in st.session_state:
        st.session_state.agent_deps = get_agent_deps()  

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    col1, col2 = st.columns(2)

    # Column 1: PDF Upload via file
    with col1:
        st.subheader("📄 Upload PDF")
        uploaded_file = st.file_uploader("Upload a PDF document from your device:", type=["pdf"])
        if st.button("📥 Add PDF to Database"):
            if uploaded_file is not None:
                with st.spinner("Processing the PDF and updating the database..."):
                    try:
                        # Read the PDF bytes
                        pdf_bytes = uploaded_file.read()
                        # Ingest the PDF into ChromaDB
                        result = asyncio.get_event_loop().run_until_complete(
                            ingest_pdf_file_to_chromadb(pdf_bytes, st.session_state.agent_deps)
                        )
                        st.success(result)
                        st.session_state.pdf_uploaded = True
                    except Exception as e:
                        st.error(f"Error processing the PDF: {e}")
            else:
                st.warning("Please upload a PDF file.")

    # Column 2: Ask a Question
    with col2:
        st.subheader("❓ Ask a Question")
        question = st.text_input("Enter your question:", placeholder="What are the responsibilities of a full-stack developer?")
        if st.button("🔍 Get Answer"):
            if question:
                with st.spinner("Thinking..."):
                    try:
                        # Always use run_rag_agent, which retrieves context from ChromaDB (including uploaded PDFs)
                        # Modify get_answer to return both answer and context
                        from rag_agent import run_rag_agent
                        answer, context = run_async(run_rag_agent(question, n_results=5, return_context=True))
                        st.success("Here's the answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error getting the answer: {e}")
            else:
                st.warning("Please enter a valid question.")

    st.markdown("---")


if __name__ == "__main__":
    asyncio.run(main())
