# AI RAG Knowledge Assistant

A practical **Retrieval-Augmented Generation (RAG)** application for enterprise knowledge-based Q&A.

This project demonstrates how to ground large language model (LLM) responses on internal documents to reduce hallucinations and improve answer reliability.

---

## Why this project

In enterprise scenarios, LLMs cannot directly access private or internal knowledge.
Pure LLM generation often leads to hallucinations or unverifiable answers.

RAG (Retrieval-Augmented Generation) solves this by:
- Retrieving relevant context from a knowledge base
- Forcing the LLM to answer based on retrieved evidence

This project focuses on **engineering-level RAG implementation**, not model training.

---

## What it does

- Ingests local documents (`.md`, `.txt`)
- Chunks documents into semantic segments
- Builds a vector index using **FAISS**
- Retrieves Top-K relevant chunks for a query
- Generates answers grounded on retrieved context
- Provides:
  - **Web UI (Streamlit)**
  - **CLI tools**
- Works **without API key** using deterministic local fallback logic

---

## Architecture

