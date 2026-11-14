"""Evaluate all four systems on the 25 labelled queries."""
import json
from pathlib import Path

from bm25 import BM25
from dense import DenseRetriever
from evaluate import run_evaluation, recall_at_k
from fusion import fuse
from pipeline import HybridRetriever


class FusedRetriever:
    """BM25 + dense -> RRF fusion, no reranking (ablation)."""
    def __init__(self):
        self.bm25 = BM25()
        self.dense = DenseRetriever()

    def fit(self, docs):
        self.bm25.fit(docs)
        self.dense.fit(docs)
        return self

    def search(self, query, k=5):
        b = [d for d, _ in self.bm25.search(query, k=20)]
        d = [d for d, _ in self.dense.search(query, k=20)]
        return fuse([b, d])[:k]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    corpus = {}
    for line in (root / "data" / "corpus.jsonl").open():
        d = json.loads(line)
        corpus[d["id"]] = d["text"]

    systems = {
        "BM25": BM25().fit(corpus),
        "Dense": DenseRetriever().fit(corpus),
        "Fused": FusedRetriever().fit(corpus),
        "Hybrid": HybridRetriever().fit(corpus),
    }
    print(f"{'system':<8} {'Recall@5':>9} {'MRR':>6} {'nDCG@5':>8}")
    for name, retriever in systems.items():
        r = run_evaluation(retriever, root / "data" / "queries.json", k=5)
        print(f"{name:<8} {r['recall@5']:>9.3f} {r['mrr']:>6.3f} {r['ndcg@5']:>8.3f}")

    # Per-query Recall@5.
    queries = json.loads((root / "data" / "queries.json").read_text())
    print(f"\n{'query':<45} {'BM25':>5} {'Dense':>6} {'Fused':>6} {'Hybrid':>7}")
    for q in queries:
        row = []
        for retriever in systems.values():
            got = [doc_id for doc_id, _ in retriever.search(q["query"], k=5)]
            row.append(recall_at_k(got, set(q["relevant"]), k=5))
        print(f"{q['query'][:44]:<45} {row[0]:>5.2f} {row[1]:>6.2f} {row[2]:>6.2f} {row[3]:>7.2f}")
