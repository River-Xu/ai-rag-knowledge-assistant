# AI RAG Knowledge Assistant

A practical **Retrieval-Augmented Generation (RAG)** application for enterprise-style knowledge Q&A.

It supports:
- Document ingestion (Markdown / TXT)
- Chunking + vector indexing (FAISS)
- Retrieval-grounded answering (LLM optional)
- Streamlit Web demo + CLI indexing/search

> If you do not configure an API key, the app still works using a **local fallback answerer** (extractive summary from retrieved context).

## Why this project
LLMs can hallucinate when asked about private/internal knowledge. RAG grounds answers on your documents by retrieving relevant context first.

## Architecture
Documents → Chunking → Embeddings → FAISS Index → Retrieve Top-K Context → (Optional) LLM Answer → UI/CLI

## Quick Start (Windows)
```bash
cd ai-rag-knowledge-assistant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

## Environment variables (optional)
Copy `.env.example` to `.env` and fill values:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)
- `OPENAI_MODEL` (optional)

## CLI usage
```bash
# build index from ./data
python -m rag.cli index --data_dir data --index_dir .rag_index

# ask question from the built index
python -m rag.cli ask --index_dir .rag_index --question "What is the login requirement?"
```

## Project Positioning
This is an **AI application engineering** project focusing on RAG integration and reliable outputs rather than model training.
