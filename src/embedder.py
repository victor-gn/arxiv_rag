import os
from typing import List
from langchain_openai import OpenAIEmbeddings
import numpy as np
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

# Inicializa o modelo de embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


def get_openai_embeddings(texts: List[str], batch_size: int = 200) -> List[List[float]]:
    """
    Cria embeddings para uma lista de textos usando o modelo OpenAI.
    Processa em batches para respeitar o limite de tokens da API.

    Args:
        texts: Lista de textos para criar embeddings
        batch_size: Tamanho de cada batch para processamento

    Returns:
        Lista de embeddings (vetores)
    """
    if not texts:
        return []

    all_embeddings = []

    # Processa os textos em batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Criando embeddings"):
        batch = texts[i : i + batch_size]
        try:
            # Cria embeddings para o batch atual
            batch_embeddings = embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Erro ao processar batch {i // batch_size + 1}: {str(e)}")
            # Se falhar, tenta processar com batch menor
            if batch_size > 10:
                print(f"Tentando com batch menor ({batch_size // 2})...")
                return get_openai_embeddings(texts, batch_size=batch_size // 2)
            else:
                raise e

    return all_embeddings


if __name__ == "__main__":
    # Exemplo de uso
    chunks = ["Texto do chunk 1...", "Texto do chunk 2...", "Texto do chunk 3..."]
    embeddings = get_openai_embeddings(chunks)
    print(f"Gerados {len(embeddings)} embeddings. Dimensão: {len(embeddings[0])}")
