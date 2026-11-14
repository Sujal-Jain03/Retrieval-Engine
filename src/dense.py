"""Dense retrieval: all-MiniLM-L6-v2 embeddings, L2-normalized, stored as
one numpy matrix (n_docs x 384). Cosine similarity is a single matmul
because all vectors are unit length — no FAISS, no vector DB.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.doc_ids: list[str] = []
        self.matrix: np.ndarray | None = None
        self.model = None


    def fit(self, docs: dict[str, str]) -> "DenseRetriever":
        """Embed all docs, normalize, store as one matrix."""
        self.model = SentenceTransformer(self.model_name)
        self.doc_ids = list(docs.keys())
        texts = [docs[doc_id] for doc_id in self.doc_ids]
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        self.matrix = emb / norms
        return self


    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (doc_id, score) pairs, best first."""
        q = self.model.encode([query], convert_to_numpy=True)[0]
        q = q / np.linalg.norm(q)
        scores = self.matrix @ q
        top = np.argsort(scores)[::-1][:k]
        return [(self.doc_ids[i], float(scores[i])) for i in top]

if __name__ == "__main__":
    import json
    from pathlib import Path

    corpus = {}
    for line in (Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl").open():
        d = json.loads(line)
        corpus[d["id"]] = d["text"]
    retriever = DenseRetriever().fit(corpus)
    print("matrix shape:", retriever.matrix.shape)
    for q in ["fresher data analyst hyderabad", "machine learning internship stipend", "jobs in accenture for freshers", "paid internship stipend per month", "fresher data analyst jobs in Hyderabad", "entry level jobs for new graduates", "monthly pay for interns", "jobs you can do from home", "26009713"]:
        print(f"\nQ: {q}")
        for doc_id, score in retriever.search(q, k=3):
            print(f"  {score:6.3f}  {doc_id}")


