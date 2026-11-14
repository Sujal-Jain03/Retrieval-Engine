"""BM25 from scratch (Robertson & Zaragoza), k1=1.5, b=0.75.

score(D, Q) = sum IDF(q) * f(q,D)*(k1+1) / (f(q,D) + k1*(1 - b + b*|D|/avgdl))
IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))

fit() builds the inverted index {term: {doc_id: count}}; search() scores
only docs containing a query term.
"""

from tokenizer import tokenize
import math


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.index: dict[str, dict[str, int]] = {}
        self.doc_len: dict[str, int] = {}
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    def fit(self, docs: dict[str, str]) -> "BM25":
        """docs: {doc_id: text}. Build the inverted index."""
        self.n_docs = len(docs)
        total_len = 0
        for doc_id, text in docs.items():
            tokens = tokenize(text)
            self.doc_len[doc_id] = len(tokens)
            total_len += len(tokens)
            counts: dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            for term, count in counts.items():
                self.index.setdefault(term, {})[doc_id] = count
        self.avgdl = total_len / self.n_docs if self.n_docs else 0.0
        return self



    def _idf(self, term: str) -> float:
        n = len(self.index.get(term, {}))
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))


    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (doc_id, score) pairs, best first."""
        scores: dict[str, float] = {}
        for term in set(tokenize(query)):
            idf = self._idf(term)
            for doc_id, f in self.index.get(term, {}).items():
                dl = self.doc_len[doc_id]
                tf_part = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                scores[doc_id] = scores.get(doc_id, 0) + idf * tf_part
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]



if __name__ == "__main__":
    import json
    from pathlib import Path

    corpus = {}
    for line in (Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl").open():
        d = json.loads(line)
        corpus[d["id"]] = d["text"]
    bm25 = BM25().fit(corpus)
    for q in ["fresher data analyst hyderabad", "machine learning internship stipend","jobs in accenture for freshers","xyznonexistentword","paid internship stipend per month","fresher data analyst jobs in Hyderabad","entry level jobs for new graduates","monthly pay for interns","jobs you can do from home","26009713"]:
        print(f"\nQ: {q}")
        for doc_id, score in bm25.search(q, k=3):
            print(f"  {score:7.2f}  {doc_id}")

    print("n_docs:", bm25.n_docs, "avgdl:", round(bm25.avgdl, 1), "terms:", len(bm25.index))