import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.ingestion.loader import load_markdown_files
from src.ingestion.chunker import chunk_documents
from src.ingestion.embeddings import get_embedding_model, create_chroma_client, build_index


def main() -> None:
    articles_dir = Path(os.getenv("ARTICLES_DIR", "data/raw/articles"))
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")

    print(f"Loading articles from: {articles_dir}")
    documents = load_markdown_files(articles_dir)
    print(f"Loaded {len(documents)} articles")

    print("Chunking documents...")
    chunks = chunk_documents(documents, chunk_size=512, chunk_overlap=64)
    print(f"Created {len(chunks)} chunks")

    print(f"Loading embedding model: {model_name}")
    model = get_embedding_model(model_name)

    print(f"Building ChromaDB index at: {persist_dir}")
    client = create_chroma_client(persist_dir)
    build_index(chunks, model, client)

    print("Done. Index is ready.")


if __name__ == "__main__":
    main()
