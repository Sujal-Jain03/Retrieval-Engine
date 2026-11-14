"""Hybrid pipeline: BM25 + dense -> RRF fusion -> cross-encoder rerank."""
import json
from pathlib import Path

from bm25 import BM25
from dense import DenseRetriever
from fusion import fuse
from reranker import Reranker


class HybridRetriever:
    def __init__(self):
        self.bm25 = BM25()
        self.dense = DenseRetriever()
        self.reranker = Reranker()
        self.docs: dict[str, str] = {}

    def fit(self, docs: dict[str, str]) -> "HybridRetriever":
        self.docs = docs
        self.bm25.fit(docs)
        self.dense.fit(docs)
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        b = [d for d, _ in self.bm25.search(query, k=20)]
        d = [d for d, _ in self.dense.search(query, k=20)]
        fused = fuse([b, d])
        top20 = [doc_id for doc_id, _ in fused[:20]]
        return self.reranker.rerank(query, top20, self.docs, k=k)


if __name__ == "__main__":
    corpus = {}
    for line in (Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl").open():
        d = json.loads(line)
        corpus[d["id"]] = d["text"]
    hybrid = HybridRetriever().fit(corpus)
    for q in ["fresher data analyst hyderabad",
              "machine learning internship stipend",
              "paid internship stipend per month",
              "26009713"]:
        print(f"\n===== Q: {q} =====")
        b = hybrid.bm25.search(q, k=3)
        print("BM25 top-3:", [doc_id for doc_id, _ in b])
        d = hybrid.dense.search(q, k=3)
        print("Dense top-3:", [doc_id for doc_id, _ in d])
        fused = fuse([[x[0] for x in hybrid.bm25.search(q, k=20)],
                      [x[0] for x in hybrid.dense.search(q, k=20)]])
        print("Fused top-5:")
        for doc_id, s in fused[:5]:
            print(f"  {s:.4f}  {doc_id}")
        print("Reranked top-5:")
        for doc_id, s in hybrid.search(q, k=5):
            print(f"  {s:6.3f}  {doc_id}")
