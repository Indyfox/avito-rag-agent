import os
import sys

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.embeddings import (
    get_embedding_model,
    create_chroma_client,
    build_index,
    load_collection,
)
from src.retrieval.retriever import HybridRetriever
from src.generation.rag_pipeline import RAGPipeline


def build(persist_dir: str, articles_dir: str, model_name: str) -> None:
    from src.ingestion.loader import load_markdown_files
    from src.ingestion.chunker import chunk_documents

    print("Loading .md files...")
    docs = load_markdown_files(Path(articles_dir))
    print(f"Loaded {len(docs)} articles")

    print("Chunking...")
    chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=64)
    print(f"Created {len(chunks)} chunks")

    print(f"Loading embedding model: {model_name}")
    model = get_embedding_model(model_name)

    print(f"Building ChromaDB index at: {persist_dir}")
    client = create_chroma_client(persist_dir)
    build_index(chunks, model, client)
    print("Index build complete.")


def chat() -> None:
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    print(f"Loading embedding model: {model_name}")
    model = get_embedding_model(model_name)
    collection = load_collection(persist_dir=chroma_dir)
    retriever = HybridRetriever(collection=collection, model=model, top_k=10)
    pipeline = RAGPipeline(retriever=retriever, llm_model=llm_model)

    print("\nAI-ассистент Авито готов. Задайте вопрос (или /exit)\n")

    while True:
        question = input("> ").strip()
        if question.lower() in ("/exit", "/quit"):
            print("До свидания!")
            break
        if not question:
            continue

        result = pipeline.query(question)
        print(f"\n{result['answer']}\n")
        if result["sources"]:
            print("---")
            for s in result["sources"]:
                print(f"  [{s['relevance']:.2f}] {s['title']} — {s['url']}")
            print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Avito RAG Agent")
    sub = parser.add_subparsers(dest="command")

    build_parser = sub.add_parser("build", help="Build ChromaDB index")
    build_parser.add_argument(
        "--articles-dir",
        default="data/raw/articles",
        help="Directory with .md articles",
    )

    sub.add_parser("chat", help="Start CLI chat")
    sub.add_parser("api", help="Start FastAPI server")

    args = parser.parse_args()

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

    if args.command == "build":
        build(persist_dir, args.articles_dir, model_name)
    elif args.command == "chat":
        chat()
    elif args.command == "api":
        import uvicorn
        uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        parser.print_help()
