# arxiv_rag

Um sistema RAG (Retrieval-Augmented Generation) para pesquisa científica, que utiliza artigos do arXiv como base de conhecimento, Qdrant como vector store e LangGraph para orquestração do fluxo.

## Descrição

Este projeto permite buscar, indexar e consultar artigos científicos do arXiv para responder perguntas sobre temas de pesquisa, combinando busca semântica (OpenAI Embeddings + Qdrant) e geração de respostas com LLMs. O fluxo é orquestrado com LangGraph, permitindo estratégias adaptativas e corretivas de RAG.

## Funcionalidades

- Download automático de artigos do arXiv por tema
- Extração de texto de PDFs
- Chunking e geração de embeddings via OpenAI
- Indexação e busca semântica com Qdrant
- Orquestração do pipeline RAG com LangGraph
- Interface web (Streamlit) para interação

## Estrutura do Projeto

```
arxiv_rag/
│
├── data/                        # PDFs e textos baixados
├── src/
│   ├── arxiv_downloader.py      # Download de artigos do arXiv
│   ├── pdf_extractor.py         # Extração de texto dos PDFs
│   ├── chunker.py               # Divisão do texto em chunks
│   ├── embedder.py              # Geração de embeddings (OpenAI)
│   ├── vectorstore.py           # Integração com Qdrant
│   ├── rag_graph.py             # Orquestração do fluxo RAG com LangGraph
│   └── interface.py             # Interface CLI ou web
│
├── docker-compose.yml           # Para subir o Qdrant
├── README.md
```

## Como rodar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/arxiv_rag.git
cd arxiv_rag
```

### 2. Crie o arquivo `.env`

```env
OPENAI_API_KEY=coloque_sua_chave_aqui
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Suba o Qdrant

```bash
docker-compose up -d
```

### 4. Instale as dependências

```bash
uv venv
source .venv/bin/activate
uv init
uv pip install -r requirements.txt  # ou use uv add ... para instalar as dependências
```

### 5. Execute a interface

```bash
streamlit run src/interface.py
```

## Tecnologias utilizadas

- [LangChain](https://python.langchain.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Qdrant](https://qdrant.tech/)
- [OpenAI API](https://platform.openai.com/)
- [arXiv API](https://arxiv.org/help/api/)
- [Streamlit](https://streamlit.io/)

## Créditos

Inspirado em tutoriais e exemplos oficiais de LangGraph, Qdrant e OpenAI.

## Licença

Este projeto está sob a licença MIT.

---

> Dicas de boas práticas para README: [Make a README](https://www.makeareadme.com/) | [freeCodeCamp: How to Write a Good README File](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)
