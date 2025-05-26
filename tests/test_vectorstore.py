import pytest
from unittest.mock import Mock, patch
from src.vectorstore import upsert_embeddings_qdrant
import uuid

@pytest.fixture
def mock_qdrant_client():
    """Fixture que mocka o cliente Qdrant"""
    with patch('src.vectorstore.QdrantClient') as mock_client:
        mock_instance = Mock()
        mock_instance.get_collections.return_value = Mock(collections=[])
        mock_client.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def sample_embeddings():
    """Fixture que retorna embeddings de exemplo"""
    return [[0.1, 0.2, 0.3] for _ in range(3)]

@pytest.fixture
def sample_metadatas():
    """Fixture que retorna metadados de exemplo"""
    return [
        {"text": f"chunk {i}", "source": f"arxiv:test{i}"}
        for i in range(3)
    ]

def test_upsert_embeddings_new_collection(mock_qdrant_client, sample_embeddings, sample_metadatas):
    """Testa a criação de uma nova collection e inserção de embeddings"""
    # Configura o mock para simular uma collection vazia
    mock_qdrant_client.get_collections.return_value = Mock(collections=[])
    
    # Testa a inserção
    upsert_embeddings_qdrant(
        embeddings=sample_embeddings,
        metadatas=sample_metadatas,
        collection_name="test_collection",
        vector_size=3,
        batch_size=2
    )
    
    # Verifica se a collection foi criada
    mock_qdrant_client.create_collection.assert_called_once()
    
    # Verifica se os pontos foram inseridos
    assert mock_qdrant_client.upsert.call_count == 2  # 2 batches de 2 e 1

def test_upsert_embeddings_existing_collection(mock_qdrant_client, sample_embeddings, sample_metadatas):
    """Testa a inserção em uma collection existente"""
    # Configura o mock para simular uma collection existente
    mock_collection = Mock()
    mock_collection.name = "test_collection"
    mock_qdrant_client.get_collections.return_value = Mock(collections=[mock_collection])
    
    # Testa a inserção
    upsert_embeddings_qdrant(
        embeddings=sample_embeddings,
        metadatas=sample_metadatas,
        collection_name="test_collection",
        vector_size=3
    )
    
    # Verifica que a collection não foi criada novamente
    mock_qdrant_client.create_collection.assert_not_called()
    
    # Verifica se os pontos foram inseridos
    assert mock_qdrant_client.upsert.call_count > 0

def test_upsert_embeddings_retry(mock_qdrant_client, sample_embeddings, sample_metadatas):
    """Testa o mecanismo de retry em caso de falha"""
    # Configura o mock para falhar duas vezes e depois funcionar
    mock_qdrant_client.upsert.side_effect = [
        Exception("First failure"),
        Exception("Second failure"),
        None  # Sucesso na terceira tentativa
    ]
    
    # Testa a inserção
    upsert_embeddings_qdrant(
        embeddings=sample_embeddings,
        metadatas=sample_metadatas,
        collection_name="test_collection",
        vector_size=3,
        batch_size=1
    )
    
    # Verifica se houve 3 tentativas
    assert mock_qdrant_client.upsert.call_count == 3

def test_upsert_embeddings_unique_ids(mock_qdrant_client, sample_embeddings, sample_metadatas):
    """Testa se os IDs gerados são únicos"""
    # Configura o mock para capturar os pontos inseridos
    inserted_points = []
    def capture_points(*args, **kwargs):
        inserted_points.extend(kwargs['points'])
        return None
    mock_qdrant_client.upsert.side_effect = capture_points
    
    # Testa a inserção
    upsert_embeddings_qdrant(
        embeddings=sample_embeddings,
        metadatas=sample_metadatas,
        collection_name="test_collection",
        vector_size=3
    )
    
    # Verifica se os IDs são únicos
    ids = [point.id for point in inserted_points]
    assert len(ids) == len(set(ids))
    
    # Verifica se os IDs são UUIDs válidos
    for id_str in ids:
        uuid.UUID(id_str)  # Isso vai falhar se não for um UUID válido 