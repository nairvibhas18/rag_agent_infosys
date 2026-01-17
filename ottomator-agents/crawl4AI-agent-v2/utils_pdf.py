import httpx
import fitz  # PyMuPDF
from typing import List, Dict, Any
from chromadb import Collection

async def download_pdf(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content

def extract_text_from_pdf(pdf_content: bytes) -> str:
    document = fitz.open(stream=pdf_content, filetype="pdf")
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def split_text_into_chunks(text: str, max_words: int = 400, overlap: float = 0.2) -> List[str]:
    words = text.split()
    chunks = []
    step_size = int(max_words * (1 - overlap))
    for start in range(0, len(words), step_size):
        end = start + max_words
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks

async def ingest_pdf_to_chromadb(
    url: str,
    collection: Collection,
    embedding_function,
    metadata: Dict[str, Any] = None,
):
    pdf_content = await download_pdf(url)
    text = extract_text_from_pdf(pdf_content)
    chunks = split_text_into_chunks(text)
    # Generate unique IDs for each chunk
    ids = [f"{url}__{i}" for i in range(len(chunks))]
    metadatas = [metadata or {"source": url}] * len(chunks)
    # Embed and add to ChromaDB
    embeddings = embedding_function(chunks)
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
