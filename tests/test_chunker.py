import pytest
from src.chunker import create_chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter


@pytest.fixture
def sample_text():
    """Fixture que retorna um texto de exemplo para chunking"""
    return """
    Este é um texto de exemplo para testar o chunking.
    Ele contém várias linhas e parágrafos diferentes.
    
    Aqui está um novo parágrafo com mais conteúdo.
    Vamos ver como o chunking funciona com textos mais longos.
    
    E aqui está outro parágrafo para testar a divisão.
    """


@pytest.fixture
def sample_metadata():
    """Fixture que retorna metadados de exemplo"""
    return {"source": "test.pdf", "page": 1, "author": "Test Author"}


def test_create_chunks(sample_text, sample_metadata):
    """Testa a criação de chunks básica"""
    chunks = create_chunks(sample_text, sample_metadata)

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    # Verifica a estrutura de cada chunk
    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert "text" in chunk
        assert "metadata" in chunk
        assert isinstance(chunk["text"], str)
        assert isinstance(chunk["metadata"], dict)
        assert len(chunk["text"]) > 0


def test_create_chunks_with_custom_splitter(sample_text, sample_metadata):
    """Testa a criação de chunks com um splitter personalizado"""
    custom_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

    chunks = create_chunks(sample_text, sample_metadata, text_splitter=custom_splitter)

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    # Verifica se os chunks respeitam o tamanho máximo
    for chunk in chunks:
        assert len(chunk["text"]) <= 50


def test_create_chunks_empty_text(sample_metadata):
    """Testa a criação de chunks com texto vazio"""
    chunks = create_chunks("", sample_metadata)
    assert len(chunks) == 0


def test_create_chunks_minimal_metadata(sample_text):
    """Testa a criação de chunks com metadados mínimos"""
    minimal_metadata = {"source": "test.pdf"}
    chunks = create_chunks(sample_text, minimal_metadata)

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    # Verifica se os metadados foram preservados
    for chunk in chunks:
        assert chunk["metadata"]["source"] == "test.pdf"


def test_create_chunks_preserves_metadata(sample_text, sample_metadata):
    """Testa se os metadados são preservados em todos os chunks"""
    chunks = create_chunks(sample_text, sample_metadata)

    for chunk in chunks:
        assert chunk["metadata"] == sample_metadata
