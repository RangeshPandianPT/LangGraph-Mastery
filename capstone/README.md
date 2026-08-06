# LangGraph Capstone Project

A comprehensive multi-agent research application powered by LangGraph, FastAPI, and React. It features time-travel (state checkpointing), real-time streaming, Local RAG (ChromaDB), and a secure code execution sandbox.

## Architecture

This project utilizes a supervisor-worker architecture:
- **Supervisor**: Routes requests between research, writing, evaluating, and fact-checking.
- **Researcher (Map-Reduce)**: Fans out tasks to a Web Researcher (Tavily/Wiki), Document Retriever (Local RAG), and Data Analyst (Python REPL).
- **Writer**: Drafts reports based on gathered contexts.
- **Fact Checker & Evaluator**: Review and critique the draft iteratively.

## Prerequisites
- Docker & Docker Compose
- Node.js (for local frontend dev)
- Python 3.10+ (for local backend dev)

## Setup

1. Copy `.env.example` to `.env` and configure your API keys (e.g., `TAVILY_API_KEY`).
2. Run the full stack via Docker Compose:
   ```bash
   docker-compose up --build -d
   ```
3. Access the application:
   - Frontend: `http://localhost:5173`
   - Backend API Docs: `http://localhost:8000/docs`
   - Ollama API: `http://localhost:11434`

## Ingesting Local Documents (RAG)
To add documents for the `document_retriever` agent:
1. Place `.txt` or `.pdf` files inside `data/raw_docs/`.
2. Run the ingestion script:
   ```bash
   python ingest.py --source data/raw_docs --persist data/chroma_db
   ```

## Testing
Run the automated test suite to verify graph routing:
```bash
pytest test_graph.py
```
