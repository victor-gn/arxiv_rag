import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock, Mock
from src import arxiv_downloader, pdf_extractor
from src.arxiv_downloader import search_arxiv, download_pdf, download_by_query


@pytest.fixture
def fake_arxiv_result():
    mock_result = MagicMock()
    mock_result.get_short_id.return_value = "1234.5678"
    mock_result.title = "Fake Paper"
    mock_result.authors = [MagicMock(name="Author", **{"name": "John Doe"})]
    mock_result.summary = "Resumo fake."
    mock_result.published = "2024-01-01"
    mock_result.entry_id = "http://arxiv.org/abs/1234.5678"

    # Simula o download do PDF criando um arquivo temporário
    def fake_download_pdf(filename):
        with open(filename, "wb") as f:
            f.write(b"%PDF-1.4\nFake PDF content\n%%EOF")

    mock_result.download_pdf.side_effect = fake_download_pdf
    return mock_result


@pytest.fixture
def temp_pdf_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<<>>\n%%EOF"
        )
        tmp_path = tmp.name
    yield tmp_path
    os.remove(tmp_path)


def test_search_arxiv_with_mock(fake_arxiv_result):
    with patch("arxiv.Search") as mock_search:
        mock_search.return_value.results.return_value = [fake_arxiv_result]
        results = arxiv_downloader.search_arxiv("fake query", max_results=1)
        assert len(results) == 1
        assert results[0].title == "Fake Paper"


def test_download_pdf_and_extract_text_with_mock(fake_arxiv_result, temp_pdf_file):
    # Mocka search_arxiv para retornar o fake_arxiv_result
    with patch("src.arxiv_downloader.search_arxiv", return_value=[fake_arxiv_result]):
        # Mocka download_pdf para copiar o temp_pdf_file
        with patch("src.arxiv_downloader.download_pdf") as mock_download_pdf:
            mock_download_pdf.side_effect = lambda result, out_dir=None: temp_pdf_file
            papers = arxiv_downloader.download_by_query("fake query", max_results=1)
            assert len(papers) == 1
            pdf_path = papers[0]["pdf_path"]
            assert os.path.exists(pdf_path)
            texto = pdf_extractor.extract_text_from_pdf(pdf_path)
            assert isinstance(texto, str)


@pytest.fixture
def mock_paper():
    """Fixture que retorna um paper mock para testes"""
    return {
        "id": "test123",
        "title": "Test Paper",
        "authors": ["Author 1", "Author 2"],
        "summary": "Test summary",
        "pdf_url": "http://arxiv.org/pdf/test123.pdf",
        "published": "2024-01-01",
        "updated": "2024-01-01",
    }


@pytest.fixture
def mock_arxiv_client():
    """Fixture que mocka o cliente arXiv"""
    with patch("src.arxiv_downloader.Client") as mock_client:
        mock_paper = Mock()
        mock_paper.entry_id = "http://arxiv.org/abs/test123"
        mock_paper.title = "Test Paper"
        mock_paper.authors = [Mock(name="Author 1"), Mock(name="Author 2")]
        mock_paper.summary = "Test summary"
        mock_paper.pdf_url = "http://arxiv.org/pdf/test123.pdf"
        mock_paper.published = "2024-01-01"
        mock_paper.updated = "2024-01-01"

        mock_instance = Mock()
        mock_instance.results.return_value = [mock_paper]
        mock_client.return_value = mock_instance
        yield mock_client


def test_search_arxiv(mock_arxiv_client):
    """Testa a função search_arxiv"""
    results = search_arxiv("test query", max_results=1)

    assert len(results) == 1
    paper = results[0]
    assert paper["id"] == "test123"
    assert paper["title"] == "Test Paper"
    assert len(paper["authors"]) == 2
    assert paper["summary"] == "Test summary"
    assert paper["pdf_url"] == "http://arxiv.org/pdf/test123.pdf"


def test_download_pdf(mock_paper, tmp_path):
    """Testa a função download_pdf"""
    # Configura o mock do paper para download
    mock_paper.download_pdf = Mock()

    # Testa download em diretório temporário
    pdf_path = download_pdf(mock_paper, str(tmp_path))

    assert pdf_path == os.path.join(str(tmp_path), "test123.pdf")
    mock_paper.download_pdf.assert_called_once_with(pdf_path)


def test_download_pdf_existing_file(mock_paper, tmp_path):
    """Testa download de PDF que já existe"""
    # Cria um arquivo fake
    pdf_path = os.path.join(str(tmp_path), "test123.pdf")
    with open(pdf_path, "w") as f:
        f.write("fake pdf content")

    # Tenta baixar novamente
    result = download_pdf(mock_paper, str(tmp_path))

    assert result == pdf_path
    mock_paper.download_pdf.assert_not_called()


@patch("src.arxiv_downloader.search_arxiv")
@patch("src.arxiv_downloader.download_pdf")
def test_download_by_query(mock_download_pdf, mock_search_arxiv, mock_paper):
    """Testa a função download_by_query"""
    # Configura os mocks
    mock_search_arxiv.return_value = [mock_paper]
    mock_download_pdf.return_value = "test123.pdf"

    # Testa o download
    pdf_paths = download_by_query("test query", max_results=1, max_workers=1)

    assert len(pdf_paths) == 1
    assert pdf_paths[0] == "test123.pdf"
    mock_search_arxiv.assert_called_once_with("test query", max_results=1)
    mock_download_pdf.assert_called_once_with(mock_paper, "data")


def test_download_by_query_error_handling(mock_arxiv_client, tmp_path):
    """Testa o tratamento de erros no download_by_query"""
    # Configura um paper que vai falhar
    mock_paper = Mock()
    mock_paper.download_pdf.side_effect = Exception("Download failed")

    # Testa o download com erro
    pdf_paths = download_by_query("test query", max_results=1, max_workers=1)

    assert len(pdf_paths) == 0  # Nenhum PDF foi baixado com sucesso


if __name__ == "__main__":
    pytest.main()
