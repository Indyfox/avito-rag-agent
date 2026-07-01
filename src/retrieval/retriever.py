from chromadb import Collection
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
        self,
        collection: Collection,
        model: SentenceTransformer,
        top_k: int = 5,
    ):
        self.collection = collection
        self.model = model
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        query_embedding = self.model.encode(
            [f"query: {query}"],
            show_progress_bar=False,
        )

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                chunks.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "score": 1.0 - results["distances"][0][i],
                    "article_id": results["metadatas"][0][i]["article_id"],
                    "title": results["metadatas"][0][i]["title"],
                    "url": results["metadatas"][0][i]["url"],
                })

        return chunks
