"""Evaluation metrics, implemented by hand: Recall@k, MRR, nDCG@k.

Binary relevance: gain is 1 for a relevant doc, 0 otherwise.
run_evaluation() takes any retriever with .search(query, k).
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def recall_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    def dcg(ids: list[str]) -> float:
        return sum(1.0 / math.log2(i + 2) for i, d in enumerate(ids[:k]) if d in relevant)

    ideal = sorted(ranked, key=lambda d: d in relevant, reverse=True)
    ideal_dcg = dcg(ideal)
    return dcg(ranked) / ideal_dcg if ideal_dcg else 0.0


def run_evaluation(retriever, queries_path: Path = ROOT / "data" / "queries.json",
                   k: int = 5) -> dict[str, float]:
    """retriever: object with .search(query, k) -> [(doc_id, score)]."""
    queries = json.loads(queries_path.read_text())
    recalls, mrrs, ndcgs = [], [], []
    for q in queries:
        ranked = [d for d, _ in retriever.search(q["query"], k=k)]
        relevant = set(q["relevant"])
        recalls.append(recall_at_k(ranked, relevant, k))
        mrrs.append(mrr(ranked, relevant))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
    n = max(len(queries), 1)
    return {
        "n_queries": len(queries),
        f"recall@{k}": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        f"ndcg@{k}": sum(ndcgs) / n,
    }


if __name__ == "__main__":
    # Toy check of the metrics (no retriever needed).
    ranked = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d2", "d4"}
    print("toy recall@5:", recall_at_k(ranked, relevant, 5))  # expect 1.0
    print("toy mrr:", mrr(ranked, relevant))                  # expect 0.5
    print("toy ndcg@5:", round(ndcg_at_k(ranked, relevant, 5), 3))
