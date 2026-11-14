import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
corpus_ids = {json.loads(line)["id"] for line in open(ROOT / "data" / "corpus.jsonl")}
queries = json.loads((ROOT / "data" / "queries.json").read_text())

print(f"corpus chunks: {len(corpus_ids)}")
print(f"queries: {len(queries)}")
problems = []
for q in queries:
    if not q["relevant"]:
        problems.append(f"NO LABELS: {q['query']}")
    for doc_id in q["relevant"]:
        if doc_id not in corpus_ids:
            problems.append(f"BAD ID {doc_id} in query: {q['query']}")
if problems:
    print("PROBLEMS:")
    for p in problems:
        print(" -", p)
else:
    print("ALL OK — Phase 1 labelling complete.")
