import os
import arxiv

from arxiv import Result
from typing import List, Any, Dict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def search_arxiv(query: str, max_results: int = 10) -> List[Result]:
    """Search for papers on arXiv based on a query.

    Args:
        query (str): The search string to find relevant papers.
        max_results (int, optional): Maximum number of results to return. Defaults to 10.

    Returns:
        List[Result]: List of arXiv results containing the found papers.

    Note:
        Results are ordered by relevance using arXiv's default criterion.
    """
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    return list(search.results())


def download_pdf(paper: Result) -> str:
    """Download a PDF from arXiv to the data directory.

    Args:
        paper (Result): arXiv paper object containing metadata and PDF URL.

    Returns:
        str: Full path to the downloaded PDF file.

    Note:
        The file is saved in DATA_DIR with the paper ID as filename.
        The paper ID is converted to a filesystem-safe format.
    """
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    paper_id = paper.get_short_id().replace("/", "_")
    # Define file path
    pdf_path = os.path.join(DATA_DIR, f"{paper_id}.pdf")

    # Download PDF
    paper.download_pdf(dirpath=DATA_DIR, filename=f"{paper_id}.pdf")

    return pdf_path


def download_by_query(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """Search and download papers from arXiv based on a query.

    Args:
        query (str): The search string to find relevant papers.
        max_results (int, optional): Maximum number of papers to download. Defaults to 20.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing paper metadata and PDF path.
            Each dictionary contains:
            - id (str): Short paper ID
            - title (str): Paper title
            - authors (List[str]): List of author names
            - summary (str): Paper abstract
            - pdf_path (str): Path to downloaded PDF file
            - published (datetime): Publication date
            - url (str): arXiv paper URL

    Note:
        PDFs are automatically downloaded to DATA_DIR.
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
