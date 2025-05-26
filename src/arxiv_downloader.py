import os
import arxiv
from typing import List, Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def search_arxiv(query: str, max_results: int = 10) -> List[Any]:
    """
    Busca artigos no arXiv por uma query e retorna uma lista de resultados.
    """
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    return list(search.results())


def download_pdf(paper: Any) -> str:
    """
    Baixa o PDF de um resultado do arXiv para o diretório data.
    """
    # Cria o diretório data se não existir
    os.makedirs(DATA_DIR, exist_ok=True)

    paper_id = paper.get_short_id().replace("/", "_")
    # Define o caminho do arquivo
    pdf_path = os.path.join(DATA_DIR, f"{paper_id}.pdf")

    # Baixa o PDF
    paper.download_pdf(dirpath=DATA_DIR, filename=f"{paper_id}.pdf")

    return pdf_path


def download_by_query(query, max_results=5):
    """
    Busca artigos e baixa os PDFs para a pasta de dados.
    Retorna uma lista de dicionários com metadados e caminho do PDF.
    """
    results = search_arxiv(query, max_results)
    papers = []
    for result in results:
        pdf_path = download_pdf(result)
        papers.append(
            {
                "id": result.get_short_id(),
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "summary": result.summary,
                "pdf_path": pdf_path,
                "published": result.published,
                "url": result.entry_id,
            }
        )
    return papers


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "transformers NLP"
    print(f"Buscando artigos para: {query}")
    papers = download_by_query(query, max_results=3)
    for paper in papers:
        print(f"- {paper['title']} ({paper['pdf_path']})")
