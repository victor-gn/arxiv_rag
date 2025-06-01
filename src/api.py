import fastapi
from fastapi.responses import RedirectResponse

from pydantic import BaseModel, Field

from src.ingest_pdf import stream_graph_updates as ingest_pdf
from src.rag_qa import stream_qa_updates as answer_question


class Query(BaseModel):
    query: str = Field(description="The query to search for")

class Response(BaseModel):
    response: str = Field(description="The response to the query")


app = fastapi.FastAPI()


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


@app.post("/ingest")
def ingest(query: Query):
    ingest_pdf(query.query)


@app.post("/answer/")
def answer(query: Query) -> Response:
    return Response(response=answer_question(query.query))
