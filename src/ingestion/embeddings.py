import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def get_embedding_model(model_name: str = "intfloat/multilingual-e5-small") -> SentenceTransformer:
    return SentenceTransformer(model_name, trust_remote_code=True)


def create_chroma_client(persist_dir: str = "./chroma_db") -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=persist_dir)


def build_index(
    chunks: list[dict],
    model: SentenceTransformer,
    client: chromadb.PersistentClient,
    collection_name: str = "avito_support",
) -> None:
    texts = [f"passage: {c['text']}" for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        ids = [f"{c['article_id']}_{c['chunk_index']}" for c in batch_chunks]
        metadatas = [
            {
                "article_id": c["article_id"],
                "title": c["title"],
                "url": c["url"],
                "chunk_index": c["chunk_index"],
            }
            for c in batch_chunks
        ]
        batch_texts = [c["text"] for c in batch_chunks]

        collection.add(
            ids=ids,
            embeddings=batch_embeddings.tolist(),
            documents=batch_texts,
            metadatas=metadatas,
        )

    print(f"Indexed {len(chunks)} chunks in collection '{collection_name}'")


def load_collection(
    persist_dir: str = "./chroma_db",
    collection_name: str = "avito_support",
) -> chromadb.Collection:
    client = create_chroma_client(persist_dir)
    return client.get_collection(collection_name)
