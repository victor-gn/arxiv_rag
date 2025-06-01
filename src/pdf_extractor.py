import pymupdf4llm


def extract_text_from_pdf(pdf_path: str, pages: list[int] | None = None) -> str:
    """Extract text from a PDF file in Markdown format using pymupdf4llm.

    Args:
        pdf_path (str): Path to the PDF file.
        pages (list[int] | None, optional): List of pages (0-based) to extract.
            If None, extracts all pages. Defaults to None.

    Returns:
        str: Extracted text in Markdown format.
    """
    return pymupdf4llm.to_markdown(pdf_path, pages=pages)
