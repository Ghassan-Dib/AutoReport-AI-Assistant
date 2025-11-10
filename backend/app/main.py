from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.rag_pipeline import get_rag_chain

app = FastAPI(title="Consigli Conversational RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chain = get_rag_chain()


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(query: Query):
    result = chain({"question": query.question})

    return {
        "content": result["answer"],
        # "sources": [doc.metadata.get("source") for doc in result["source_documents"]],
    }
