from langchain_text_splitters.markdown import MarkdownTextSplitter

def chunk_markdown_text(markdown_text, chunk_size=1000, chunk_overlap=200):
    """
    Divide o texto Markdown em chunks respeitando a estrutura de headings.
    :param markdown_text: Texto em Markdown a ser chunkado.
    :param chunk_size: Tamanho máximo de cada chunk (em caracteres).
    :param chunk_overlap: Sobreposição entre chunks (em caracteres).
    :return: Lista de strings, cada uma representando um chunk.
    """
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(markdown_text)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python chunker.py <arquivo_markdown>")
    else:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            md = f.read()
        chunks = chunk_markdown_text(md)
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---\n{chunk[:500]}")
