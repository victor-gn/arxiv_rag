import os
import pymupdf4llm

def extract_text_from_pdf(pdf_path, pages=None):
    """
    Extrai o texto de um PDF em formato Markdown usando pymupdf4llm.
    :param pdf_path: Caminho para o arquivo PDF.
    :param pages: Lista de páginas (0-based) a extrair, ou None para todas.
    :return: Texto extraído em Markdown.
    """
    return pymupdf4llm.to_markdown(pdf_path, pages=pages)

if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not pdf_path or not os.path.exists(pdf_path):
        print("Informe o caminho de um PDF válido.")
    else:
        texto = extract_text_from_pdf(pdf_path)
        print(texto[:1000])  # Mostra os primeiros 1000 caracteres
