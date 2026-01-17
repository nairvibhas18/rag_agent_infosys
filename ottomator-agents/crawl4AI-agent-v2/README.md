# Pydantic AI Documentation Crawler & RAG Agent

An intelligent documentation crawler and retrieval-augmented generation (RAG) system, powered by Crawl4AI and Pydantic AI. This project enables you to crawl, chunk, and vectorize documentation from any website, pdf, `.txt`/Markdown pages (llms.txt), or sitemap, and interact with the knowledge base using a Streamlit interface.

---

## Features

- **Flexible documentation crawling:** Handles regular websites, `.txt`/Markdown pages (llms.txt), and sitemaps.
- **Parallel and recursive crawling:** Efficiently gathers large doc sites with memory-adaptive batching.
- **Smart chunking:** Hierarchical Markdown chunking by headers, ensuring chunks are optimal for vector search.
- **Vector database integration:** Stores chunks and metadata in ChromaDB for fast semantic retrieval.
- **Streamlit RAG interface:** Query your documentation with LLM-powered semantic search.
---

## Prerequisites

- Python 3.11+
- OpenAI API key (for embeddings and LLM-powered search)
- Crawl4AI/Playwright and other dependencies in `requirements.txt`
- Streamlit for the web interface

---

## Installation

1. **Install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   playwright install
   ```

2. **Set up environment variables:**
   - Edit `.env` with your API keys and preferences:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     MODEL_CHOICE=gpt-4o  # or your preferred OpenAI model
     ```

---

## Usage

### 1. Crawling and Inserting Documentation

The main entry point for crawling and vectorizing documentation is [`insert_docs.py`](insert_docs.py):

#### Supported URL Types

- **Regular documentation sites:** Recursively crawls all internal links, deduplicates by URL (ignoring fragments).
- **Markdown or .txt pages (such as llms.txt):** Fetches and chunks Markdown content.
- **Sitemaps (`sitemap.xml`):** Batch-crawls all URLs listed in the sitemap.

#### Example Usage

```bash
python insert_docs.py <URL> [--collection mydocs] [--db-dir ./chroma_db] [--embedding-model text-embedding-small-3] [--chunk-size 1000] [--max-depth 3] [--max-concurrent 10] [--batch-size 100]
```

**Arguments:**
- `URL`: The root URL, .txt file, or sitemap to crawl.
- `--collection`: ChromaDB collection name (default: `docs`)
- `--db-dir`: Directory for ChromaDB data (default: `./chroma_db`)
- `--embedding-model`: Embedding model for vector storage (default: `text-embedding-small-3`)
- `--chunk-size`: Maximum characters per chunk (default: `1000`)
- `--max-depth`: Recursion depth for regular URLs (default: `3`)
- `--max-concurrent`: Max parallel browser sessions (default: `10`)
- `--batch-size`: Batch size for ChromaDB insertion (default: `100`)

**Examples for each type (regular URL, .txt, sitemap):**
```bash
python insert_docs.py https://ai.pydantic.dev/
python insert_docs.py https://ai.pydantic.dev/llms-full.txt
python insert_docs.py https://ai.pydantic.dev/sitemap.xml
```

#### Chunking Strategy

- Splits content first by `#`, then by `##`, then by `###` headers.
- If a chunk is still too large, splits by character count.
- All chunks are less than the specified `--chunk-size` (default: 1000 characters).

#### Metadata

Each chunk is stored with:
- Source URL
- Chunk index
- Extracted headers
- Character and word counts

---

### 2. Running the Streamlit RAG Interface

After crawling and inserting docs, launch the Streamlit app for semantic search and question answering:

```bash
streamlit run streamlit_app.py
```

- The interface will be available at [http://localhost:8501](http://localhost:8501)
- Query your documentation using natural language and get context-rich answers.

---

## Project Structure

```
crawl4AI-agent/
├── insert_docs.py
├── rag_agent.py
├── streamlit_app.py
├── utils.py
├── requirements.txt
├── .env
└── README.md
```

---

## Advanced Usage & Customization

- **Chunking:** Tune `--chunk-size` for your retrieval use case.
- **Embeddings:** Swap out the embedding model with `--embedding-model`.
- **Crawling:** Adjust `--max-depth` and `--max-concurrent` for large sites.
- **Vector DB:** Use your own ChromaDB directory or collection for multiple projects.

---

## Troubleshooting

- Ensure all dependencies are installed and environment variables are set.
- For large sites, increase memory or decrease `--max-concurrent`.
- If you encounter crawling issues, try running the example scripts for isolated debugging.