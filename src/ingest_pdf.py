from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import List, Dict, Any
from src import arxiv_downloader, pdf_extractor, chunker, embedder
from src.vectorstore import QdrantVectorStore
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class State(TypedDict):
    """State dictionary for the RAG ingestion pipeline.

    Attributes:
        query (str): The search query for arXiv.
        arxiv_results (List[Any]): List of arXiv search results.
        pdf_paths (List[str]): List of paths to downloaded PDF files.
        markdowns (List[str]): List of extracted text in markdown format.
        chunks (List[str]): List of text chunks after splitting.
        embeddings (List[list]): List of embedding vectors.
        metadata (List[Dict[str, Any]]): List of metadata for each chunk.
    """

    query: str
    arxiv_results: List[Any]
    pdf_paths: List[str]
    markdowns: List[str]
    chunks: List[str]
    embeddings: List[list]
    metadata: List[Dict[str, Any]]


def search_arxiv_node(state: State) -> State:
    """Search for articles on arXiv based on the query.

    Args:
        state (State): Current state containing the search query.

    Returns:
        State: Updated state with arXiv search results.
    """
    results = arxiv_downloader.search_arxiv(state["query"], max_results=30)
    logger.info(f"Found {len(results)} articles on arXiv")
    return {"arxiv_results": results}


def download_pdfs_node(state: State) -> State:
    """Download PDFs for the found arXiv articles.

    Args:
        state (State): Current state containing arXiv search results.

    Returns:
        State: Updated state with paths to downloaded PDFs.
    """
    pdf_paths = [arxiv_downloader.download_pdf(r) for r in state["arxiv_results"]]
    logger.info(f"Downloaded {len(pdf_paths)} PDFs")
    return {"pdf_paths": pdf_paths}


def extract_text_node(state: State) -> State:
    """Extract text from PDFs in markdown format.

    Args:
        state (State): Current state containing PDF paths.

    Returns:
        State: Updated state with extracted markdown texts.
    """
    markdowns = [pdf_extractor.extract_text_from_pdf(p) for p in state["pdf_paths"]]
    logger.info(f"Extracted text from {len(markdowns)} PDFs")
    return {"markdowns": markdowns}


def chunking_node(state: State) -> State:
    """Split markdown texts into chunks and prepare metadata.

    Args:
        state (State): Current state containing markdown texts and arXiv results.

    Returns:
        State: Updated state with text chunks and their metadata.
    """
    chunks, metadata = [], []
    for i, md in enumerate(state["markdowns"]):
        doc_chunks = chunker.chunk_markdown_text(md)
        chunks.extend(doc_chunks)
        metadata.extend(
            [
                {
                    "arxiv_id": state["arxiv_results"][i].get_short_id(),
                    "chunk_idx": j,
                    "text": c,
                }
                for j, c in enumerate(doc_chunks)
            ]
        )
    logger.info(
        f"Generated {len(chunks)} chunks from {len(state['markdowns'])} documents"
    )
    return {"chunks": chunks, "metadata": metadata}


def embedding_node(state: State) -> State:
    """Generate embeddings for text chunks.

    Args:
        state (State): Current state containing text chunks.

    Returns:
        State: Updated state with embedding vectors.
    """
    embeddings = embedder.get_openai_embeddings(state["chunks"])
    logger.info(f"Generated {len(embeddings)} embeddings")
    return {"embeddings": embeddings}


def qdrant_node(state: State) -> State:
    """Insert embeddings and metadata into Qdrant vector store.

    Args:
        state (State): Current state containing embeddings and metadata.

    Returns:
        State: Updated state with final status.
    """
    qdrant_vector_store = QdrantVectorStore()
    qdrant_vector_store.upsert_embeddings_qdrant(
        state["embeddings"],
        state["metadata"],
        collection_name=os.getenv("QDRANT_COLLECTION", "arxiv_chunks"),
    )
    logger.info("Successfully inserted embeddings into Qdrant")
    return {}


# Build the graph
graph = StateGraph(State)
graph.add_node("search_arxiv", search_arxiv_node)
graph.add_node("download_pdfs", download_pdfs_node)
graph.add_node("extract_text", extract_text_node)
graph.add_node("chunking", chunking_node)
graph.add_node("embedding", embedding_node)
graph.add_node("qdrant", qdrant_node)

graph.add_edge(START, "search_arxiv")
graph.add_edge("search_arxiv", "download_pdfs")
graph.add_edge("download_pdfs", "extract_text")
graph.add_edge("extract_text", "chunking")
graph.add_edge("chunking", "embedding")
graph.add_edge("embedding", "qdrant")
graph.add_edge("qdrant", END)

rag_pipeline = graph.compile()


def stream_graph_updates(query: str):
    """Stream updates from the RAG pipeline execution.

    Args:
        query (str): The search query to process.

    Note:
        This function logs real-time updates about the pipeline's progress,
        including the number of articles found, PDFs downloaded, texts extracted,
        chunks generated, and embeddings created.
    """
    state = {"query": query}
    logger.info(f"Starting pipeline with query: {query}")

    for update in rag_pipeline.stream(state, stream_mode="updates"):
        node_name = list(update.keys())[0]
        logger.info(f"Node: {node_name}")

        if node_name == "search_arxiv":
            logger.info(
                f"Articles found: {len(update[node_name].get('arxiv_results', []))}"
            )
        elif node_name == "download_pdfs":
            logger.info(
                f"PDFs downloaded: {len(update[node_name].get('pdf_paths', []))}"
            )
        elif node_name == "extract_text":
            logger.info(
                f"Texts extracted: {len(update[node_name].get('markdowns', []))}"
            )
        elif node_name == "chunking":
            logger.info(f"Chunks generated: {len(update[node_name].get('chunks', []))}")
        elif node_name == "embedding":
            logger.info(
                f"Embeddings generated: {len(update[node_name].get('embeddings', []))}"
            )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2 and sys.argv[1] == "viz":
        logger.info(rag_pipeline.get_graph().draw_mermaid())
    else:
        query = sys.argv[1]
        stream_graph_updates(query)
