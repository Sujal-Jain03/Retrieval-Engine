"""Cross-encoder reranking (ms-marco-MiniLM-L-6-v2). Scores each
(query, doc) pair together, so it is more accurate and much slower than
the retrievers — it only reranks the top-20 candidates.
"""
import numpy as np
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model=None

    def rerank(self, query: str, doc_ids: list[str],
            docs: dict[str, str], k: int = 5) -> list[tuple[str, float]]:
        """Re-score candidates and return top-k (doc_id, score)."""
        if self.model is None:
            self.model = CrossEncoder(self.model_name)
        pairs = [(query, docs[doc_id]) for doc_id in doc_ids]
        scores = self.model.predict(pairs)
        order = np.argsort(scores)[::-1][:k]
        return [(doc_ids[i], float(scores[i])) for i in order]

