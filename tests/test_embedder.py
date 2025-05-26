import pytest
from unittest.mock import patch, MagicMock
from src.embedder import create_embeddings
import numpy as np


@pytest.fixture
def sample_texts():
    """Fixture que retorna textos de exemplo para embedding"""
    return [
        "Este é o primeiro texto de teste",
        "Aqui está o segundo texto",
        "E este é o terceiro texto para teste",
    ]


@pytest.fixture
def sample_embeddings():
    """Fixture que retorna embeddings de exemplo"""
    return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]


@patch("src.embedder.OpenAIEmbeddings")
def test_create_embeddings(mock_embeddings, sample_texts, sample_embeddings):
    """Testa a criação de embeddings"""
    # Configura o mock
    mock_instance = MagicMock()
    mock_instance.embed_documents.return_value = sample_embeddings
    mock_embeddings.return_value = mock_instance

    # Testa a função
    embeddings = create_embeddings(sample_texts)

    # Verifica os resultados
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(
        len(emb) == 3 for emb in embeddings
    )  # Dimensão dos embeddings de exemplo

    # Verifica se o mock foi chamado corretamente
    mock_instance.embed_documents.assert_called_once_with(sample_texts)


@patch("src.embedder.OpenAIEmbeddings")
def test_create_embeddings_empty_list(mock_embeddings):
    """Testa a criação de embeddings com lista vazia"""
    mock_instance = MagicMock()
    mock_instance.embed_documents.return_value = []
    mock_embeddings.return_value = mock_instance

    embeddings = create_embeddings([])
    assert isinstance(embeddings, list)
    assert len(embeddings) == 0


@patch("src.embedder.OpenAIEmbeddings")
def test_create_embeddings_error_handling(mock_embeddings, sample_texts):
    """Testa o tratamento de erros na criação de embeddings"""
    mock_instance = MagicMock()
    mock_instance.embed_documents.side_effect = Exception("Erro na API")
    mock_embeddings.return_value = mock_instance

    with pytest.raises(Exception) as exc_info:
        create_embeddings(sample_texts)
    assert "Erro na API" in str(exc_info.value)


@patch("src.embedder.OpenAIEmbeddings")
def test_create_embeddings_dimensions(mock_embeddings, sample_texts):
    """Testa se as dimensões dos embeddings estão corretas"""
    # Cria embeddings com dimensão 1536 (padrão do text-embedding-3-small)
    mock_embeddings_list = [np.random.rand(1536).tolist() for _ in sample_texts]

    mock_instance = MagicMock()
    mock_instance.embed_documents.return_value = mock_embeddings_list
    mock_embeddings.return_value = mock_instance

    embeddings = create_embeddings(sample_texts)

    assert len(embeddings) == len(sample_texts)
    assert all(len(emb) == 1536 for emb in embeddings)
