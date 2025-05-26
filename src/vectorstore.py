from qdrant_client import QdrantClient, models
from typing import List, Dict, Any
import time
import uuid


def upsert_embeddings_qdrant(
    embeddings: List[list],
    metadatas: List[Dict[str, Any]],
    collection_name: str = "arxiv_chunks",
    host: str = "localhost",
    port: int = 6333,
    vector_size: int = 1536,  # text-embedding-3-small
    distance=models.Distance.COSINE,
    batch_size: int = 100,
    timeout: int = 60,
):
    """
    Insere embeddings e metadados em uma collection do Qdrant em batches.
    :param embeddings: Lista de vetores (embeddings).
    :param metadatas: Lista de dicionários com metadados (payloads).
    :param collection_name: Nome da collection no Qdrant.
    :param host: Host do Qdrant.
    :param port: Porta do Qdrant.
    :param vector_size: Dimensão dos embeddings.
    :param distance: Métrica de distância.
    :param batch_size: Tamanho de cada batch para inserção.
    :param timeout: Timeout em segundos para cada operação.
    """
    # Inicializa o cliente com timeout aumentado
    client = QdrantClient(host=host, port=port, timeout=timeout)

    # Cria a collection se não existir
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )

    # Insere em batches
    total = len(embeddings)
    for i in range(0, total, batch_size):
        batch_embeddings = embeddings[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]

        # Gera IDs únicos usando UUID para cada ponto
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),  # Gera um UUID único para cada ponto
                vector=vec,
                payload=meta,
            )
            for vec, meta in zip(batch_embeddings, batch_metadatas)
        ]

        # Tenta inserir o batch com retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                )
                print(f"Batch {i // batch_size + 1} inserido com sucesso!")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(
                    f"Erro no batch {i // batch_size + 1}, tentando novamente... ({attempt + 1}/{max_retries})"
                )
                time.sleep(2)  # Espera 2 segundos antes de tentar novamente


if __name__ == "__main__":
    # Exemplo de uso
    # embeddings = [[...], ...]
    # metadatas = [{"text": chunk, "source": "arxiv:xxxx"}, ...]
    pass
