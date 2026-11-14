"""Reciprocal rank fusion (Cormack et al.): fuse rankings by rank, not
score — RRF(d) = sum 1/(k + rank), k=60. BM25 and dense scores live on
different scales, so ranks are the common currency.
"""


def fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists into one ranking. Returns (doc_id, rrf_score)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

