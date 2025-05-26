import pytest
import os
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Fixture que cria e limpa um diretório temporário para dados de teste"""
    test_dir = tmp_path_factory.mktemp("test_data")
    yield test_dir
    # Limpa o diretório após os testes
    shutil.rmtree(test_dir)


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Fixture que configura variáveis de ambiente para testes"""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")
