import numpy as np
from chromadb import Collection
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    def __init__(
        self,
        collection: Collection,
        model: SentenceTransformer,
        top_k: int = 10,
        dense_weight: float = 0.6,
    ):
        self.collection = collection
        self.model = model
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.bm25_weight = 1.0 - dense_weight

        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        results = self.collection.get(include=["documents"])
        self._all_docs = results["documents"] or []
        self._all_ids = results["ids"] or []

        if self._all_docs:
            tokenized = [self._tokenize(d) for d in self._all_docs]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().replace(",", " ").replace(".", " ").split()

    def retrieve(self, query: str) -> list[dict]:
        k = min(self.top_k * 2, len(self._all_docs))

        # Dense retrieval
        query_embedding = self.model.encode(
            [f"query: {query}"],
            show_progress_bar=False,
        )
        dense_results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        dense_scores: dict[str, float] = {}
        if dense_results["ids"] and dense_results["ids"][0]:
            for i in range(len(dense_results["ids"][0])):
                chunk_id = dense_results["ids"][0][i]
                score = 1.0 - dense_results["distances"][0][i]
                dense_scores[chunk_id] = score

        # BM25 retrieval
        bm25_scores: dict[str, float] = {}
        if self._bm25 is not None:
            tokenized_query = self._tokenize(query)
            bm25_raw = self._bm25.get_scores(tokenized_query)
            max_score = float(bm25_raw.max()) if bm25_raw.max() > 0 else 1.0
            for idx, score in enumerate(bm25_raw):
                if score > 0:
                    chunk_id = self._all_ids[idx]
                    bm25_scores[chunk_id] = score / max_score

        # Reciprocal Rank Fusion
        fused = self._rrf_fuse(dense_scores, bm25_scores, self.dense_weight, self.bm25_weight)

        # Build result list
        all_metadata = {}
        all_documents = {}
        if dense_results["metadatas"] and dense_results["metadatas"][0]:
            for i in range(len(dense_results["ids"][0])):
                cid = dense_results["ids"][0][i]
                all_metadata[cid] = dense_results["metadatas"][0][i]
                all_documents[cid] = dense_results["documents"][0][i]

        sorted_chunks = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:self.top_k]

        chunks = []
        for chunk_id, score in sorted_chunks:
            meta = all_metadata.get(chunk_id, {})
            text = all_documents.get(chunk_id, "")
            chunks.append({
                "id": chunk_id,
                "text": text,
                "score": score,
                "article_id": meta.get("article_id", ""),
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
            })

        return chunks

    @staticmethod
    def _rrf_fuse(
        dense: dict[str, float],
        bm25: dict[str, float],
        dense_w: float,
        bm25_w: float,
    ) -> dict[str, float]:
        all_ids = set(dense.keys()) | set(bm25.keys())

        if not dense and not bm25:
            return {}

        max_dense = max(dense.values()) if dense else 1.0
        max_bm25 = max(bm25.values()) if bm25 else 1.0

        fused = {}
        for cid in all_ids:
            d_score = (dense.get(cid, 0.0) / max_dense) if max_dense > 0 else 0.0
            b_score = (bm25.get(cid, 0.0) / max_bm25) if max_bm25 > 0 else 0.0
            fused[cid] = d_score * dense_w + b_score * bm25_w

        return fused

    @property
    def chunk_count(self) -> int:
        return len(self._all_docs)
