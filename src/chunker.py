from langchain_text_splitters.markdown import MarkdownTextSplitter


def chunk_markdown_text(
    markdown_text, chunk_size: int = 1000, chunk_overlap: int = 200
):
    """Split Markdown text into chunks while preserving heading structure.

    Args:
        markdown_text (str): The Markdown text to be split into chunks.
        chunk_size (int, optional): Maximum size of each chunk in characters. Defaults to 1000.
        chunk_overlap (int, optional): Number of characters to overlap between chunks. Defaults to 200.

    Returns:
        List[str]: List of strings, each representing a chunk of the original text.

    Note:
        The splitting process respects Markdown heading structure to maintain document coherence.
    """
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(markdown_text)
