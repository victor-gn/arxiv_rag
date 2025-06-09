# arXiv RAG

A Retrieval-Augmented Generation (RAG) system for answering questions about research topics, using arXiv articles as a knowledge base, Qdrant as a vector store, and LangGraph for pipeline orchestration.

## Description

This project enables searching, indexing, and querying scientific articles from arXiv to answer research questions, combining semantic search (OpenAI Embeddings + Qdrant) and answer generation with LLMs. The flow is orchestrated with LangGraph, allowing adaptive and corrective RAG strategies.

## Features

- Automatic download of arXiv articles by topic
- PDF text extraction
- Chunking and embedding generation via OpenAI
- Semantic indexing and search with Qdrant
- RAG pipeline orchestration with LangGraph
- Question answering via FastAPI interface
- Dockerized deployment with Docker Compose


## Project Structure

```
arxiv_rag/
│
├── data/                        # Downloaded PDFs and texts
├── src/
│   ├── arxiv_downloader.py      # arXiv article downloader
│   ├── pdf_extractor.py         # PDF text extraction
│   ├── chunker.py               # Text chunking
│   ├── embedder.py              # Embedding generation (OpenAI)
│   ├── vectorstore.py           # Qdrant integration
│   ├── rag_graph.py             # RAG flow orchestration with LangGraph
│   └── interface.py             # CLI or web interface
│
├── docker-compose.yml           # Docker Compose for app and Qdrant
├── Dockerfile                   # FastAPI app Dockerfile
├── README.md
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/arxiv_rag.git
cd arxiv_rag
```

### 2. Create the `.env` file

```env
OPENAI_API_KEY=your_openai_key_here
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

### 3. Start the services with Docker Compose

```bash
docker-compose up --build
```

- The FastAPI app will be available at: http://localhost:8000/docs
- Qdrant will be available at: http://localhost:6333


## API Usage

After starting the services with Docker Compose, you can access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### Available Endpoints

#### 1. Ingest Articles

![51ea8b8d-a1d1-4941-bfc9-d8f8a24c7f63](https://github.com/user-attachments/assets/65ee5b59-3ad9-4766-a3e6-dceedaecc7e5)

Adds new articles to the knowledge base.

**Request:**
```
POST /ingest
Content-Type: application/json

{
    "query": "quantum computing"
}
```

#### 2. Answer Questions

![320dbbd0-edbd-4ac9-bc72-2ec4da108328](https://github.com/user-attachments/assets/dcd2f96a-bb3c-469a-b7bc-2e8dc97ea562)

Query the knowledge base with questions about the indexed articles.

**Request:**
```
POST /answer
Content-Type: application/json

{
    "query": "What are the latest advances in quantum computing?"
}
```

**Response:**
```json
{
    "response": "Based on the indexed articles..."
}
```


## Technologies Used

- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [LangChain](https://python.langchain.com/) — LLM orchestration
- [LangGraph](https://langchain-ai.github.io/langgraph/) — RAG pipeline orchestration
- [Qdrant](https://qdrant.tech/) — Vector database
- [OpenAI API](https://platform.openai.com/) — Embeddings and LLMs
- [arXiv API](https://arxiv.org/help/api/) — Scientific articles
- [uv](https://github.com/astral-sh/uv) — Python dependency manager

## Credits

Inspired by official tutorials and examples from LangGraph, Qdrant, and OpenAI.

## License

This project is licensed under the MIT License.
