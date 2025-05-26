import pytest
from unittest.mock import patch, MagicMock
from src.pdf_extractor import extract_text_from_pdf, extract_text_from_pdf_with_pypdf
import os
import tempfile


@pytest.fixture
def sample_pdf_content():
    """Fixture que retorna conteúdo de exemplo para um PDF"""
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<<>>\n%%EOF"


@pytest.fixture
def temp_pdf_file(sample_pdf_content):
    """Fixture que cria um arquivo PDF temporário para testes"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(sample_pdf_content)
        tmp_path = tmp.name
    yield tmp_path
    os.remove(tmp_path)


def test_extract_text_from_pdf_with_pypdf(temp_pdf_file):
    """Testa a extração de texto usando PyPDF"""
    text = extract_text_from_pdf_with_pypdf(temp_pdf_file)
    assert isinstance(text, str)


def test_extract_text_from_pdf_with_pypdf_nonexistent_file():
    """Testa a extração de texto de um arquivo inexistente"""
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf_with_pypdf("nonexistent.pdf")


def test_extract_text_from_pdf_with_pypdf_invalid_pdf():
    """Testa a extração de texto de um PDF inválido"""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(b"Invalid PDF content")
        tmp.flush()
        with pytest.raises(Exception):
            extract_text_from_pdf_with_pypdf(tmp.name)


@patch("src.pdf_extractor.extract_text_from_pdf_with_pypdf")
def test_extract_text_from_pdf(mock_extract, temp_pdf_file):
    """Testa a função principal de extração de texto"""
    mock_extract.return_value = "Texto de teste"
    text = extract_text_from_pdf(temp_pdf_file)
    assert text == "Texto de teste"
    mock_extract.assert_called_once_with(temp_pdf_file)


@patch("src.pdf_extractor.extract_text_from_pdf_with_pypdf")
def test_extract_text_from_pdf_error_handling(mock_extract, temp_pdf_file):
    """Testa o tratamento de erros na extração de texto"""
    mock_extract.side_effect = Exception("Erro na extração")
    with pytest.raises(Exception) as exc_info:
        extract_text_from_pdf(temp_pdf_file)
    assert "Erro na extração" in str(exc_info.value)
