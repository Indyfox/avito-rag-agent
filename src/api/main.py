import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.ingestion.embeddings import get_embedding_model, load_collection
from src.retrieval.retriever import HybridRetriever
from src.generation.rag_pipeline import RAGPipeline
from src.api.schemas import QueryRequest, QueryResponse, SourceInfo

load_dotenv()

app = FastAPI(
    title="Avito RAG Agent API",
    description="AI-ассистент по статьям поддержки Авито",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        model = get_embedding_model(embedding_model_name)
        collection = load_collection(persist_dir=chroma_dir)
        retriever = HybridRetriever(collection=collection, model=model, top_k=10)
        _pipeline = RAGPipeline(retriever=retriever, llm_model=llm_model)

    return _pipeline


@app.on_event("startup")
async def startup() -> None:
    get_pipeline()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    pipeline = get_pipeline()
    result = pipeline.query(question=request.question, top_k=request.top_k)

    sources = [
        SourceInfo(
            title=s["title"],
            url=s["url"],
            relevance=s["relevance"],
        )
        for s in result["sources"]
    ]

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
