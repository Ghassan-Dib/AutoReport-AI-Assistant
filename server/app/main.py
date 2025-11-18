from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag_pipeline import clear_session_history, get_rag_chain

app = FastAPI(title="Consigli Conversational RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str
    session_id: str | None = "default"


class ClearHistoryRequest(BaseModel):
    session_id: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the Consigli Conversational RAG API!"}


@app.post("/ask")
def ask(query: Query):
    """
    Ask a question with conversational context.
    Each session_id maintains its own conversation history.
    """
    try:
        chain = get_rag_chain()

        # Invoke with session configuration
        result = chain.invoke(
            {"input": query.question},
            config={"configurable": {"session_id": query.session_id}},
        )

        return {
            "content": result["answer"],
            "session_id": query.session_id,
            # "sources": [doc.metadata.get("source") for doc in result.get("context", [])]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/clear-history")
def clear_history(request: ClearHistoryRequest):
    """Clear conversation history for a specific session."""
    try:
        clear_session_history(request.session_id)
        return {
            "message": f"History cleared for session: {request.session_id}",
            "session_id": request.session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")
