from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import List, Dict, Any
from src import arxiv_downloader, pdf_extractor, chunker, embedder, vectorstore
import os
from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    query: str
    arxiv_results: List[Any]
    pdf_paths: List[str]
    markdowns: List[str]
    chunks: List[str]
    embeddings: List[list]
    metadatas: List[Dict[str, Any]]
    status: str


# Node 1: Buscar artigos no arXiv
def search_arxiv_node(state: State) -> State:
    results = arxiv_downloader.search_arxiv(state["query"], max_results=10)
    return {"arxiv_results": results, "status": "arxiv_ok"}


# Node 2: Baixar PDFs
def download_pdfs_node(state: State) -> State:
    pdf_paths = [arxiv_downloader.download_pdf(r) for r in state["arxiv_results"]]
    return {"pdf_paths": pdf_paths, "status": "pdfs_ok"}


# Node 3: Extrair texto dos PDFs
def extract_text_node(state: State) -> State:
    markdowns = [pdf_extractor.extract_text_from_pdf(p) for p in state["pdf_paths"]]
    return {"markdowns": markdowns, "status": "markdown_ok"}


# Node 4: Chunking
def chunking_node(state: State) -> State:
    chunks, metadatas = [], []
    for i, md in enumerate(state["markdowns"]):
        doc_chunks = chunker.chunk_markdown_text(md)
        chunks.extend(doc_chunks)
        metadatas.extend(
            [
                {
                    "arxiv_id": state["arxiv_results"][i].get_short_id(),
                    "chunk_idx": j,
                    "text": c,
                }
                for j, c in enumerate(doc_chunks)
            ]
        )
    return {"chunks": chunks, "metadatas": metadatas, "status": "chunks_ok"}


# Node 5: Embeddings
def embedding_node(state: State) -> State:
    embeddings = embedder.get_openai_embeddings(state["chunks"])
    return {"embeddings": embeddings, "status": "embeddings_ok"}


# Node 6: Inserir no Qdrant
def qdrant_node(state: State) -> State:
    vectorstore.upsert_embeddings_qdrant(
        state["embeddings"],
        state["metadatas"],
        collection_name=os.getenv("QDRANT_COLLECTION", "arxiv_chunks"),
    )
    return {"status": "done"}


# Monta o grafo
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
    """Função para fazer streaming das atualizações do estado do grafo."""
    state = {"query": query}
    print("\nIniciando pipeline com query:", query)
    print("-" * 50)

    for update in rag_pipeline.stream(state, stream_mode="updates"):
        print(update)
        node_name = list(update.keys())[0]
        print(f"\nNode: {node_name}")
        print(f"Status: {update[node_name].get('status', 'N/A')}")
        if node_name == "search_arxiv":
            print(
                f"Artigos encontrados: {len(update[node_name].get('arxiv_results', []))}"
            )
        elif node_name == "download_pdfs":
            print(f"PDFs baixados: {len(update[node_name].get('pdf_paths', []))}")
        elif node_name == "extract_text":
            print(f"Textos extraídos: {len(update[node_name].get('markdowns', []))}")
        elif node_name == "chunking":
            print(f"Chunks gerados: {len(update[node_name].get('chunks', []))}")
        elif node_name == "embedding":
            print(f"Embeddings gerados: {len(update[node_name].get('embeddings', []))}")
        print("-" * 50)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2 and sys.argv[1] == "viz":
        print(rag_pipeline.get_graph().draw_mermaid())
        print(
            "\nCopie o código Mermaid acima e cole em https://mermaid.live para visualizar o grafo!"
        )
    elif len(sys.argv) < 2:
        print("Uso: python -m src.rag_graph <query>  ou  python -m src.rag_graph viz")
    else:
        query = sys.argv[1]
        stream_graph_updates(query)
