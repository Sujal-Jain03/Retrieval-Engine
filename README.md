# Retrieval, Built From Scratch

A retrieval engine implemented by hand — no LangChain, no vector database,
no retrieval library — tested on real, messy data.

**Why this exists:** the Auric AI "AI Researcher: Reasoning & Retrieval"
application asks for someone who has "built something that survived real,
messy data" and who understands "language models and retrieval mechanically,
not as APIs". This project is the evidence for that essay. Every core
component is implemented from first principles, and every claim in the
essay points at a real file, a real number, or a real failure mode
found in the evaluation log.

## What "from scratch" means here

| Component | Allowed | Not allowed |
|---|---|---|
| Tokenization | `re`, hand-written rules | NLTK/spaCy tokenizers |
| BM25 | pure Python, hand-written TF/IDF | rank_bm25, Elasticsearch |
| Dense retrieval | sentence-transformers for embeddings only; similarity + indexing in numpy | FAISS, Qdrant, Chroma |
| Fusion | hand-written RRF | any fusion library |
| Reranking | hand-written scorer / small cross-encoder | Cohere rerank API |
| Evaluation | hand-written Recall@k, MRR, nDCG | RAGAS, DeepEval |

Embeddings come from a pretrained model (all-MiniLM-L6-v2) — training an
embedding model from scratch is out of scope. Everything *around* the
embeddings (indexing, similarity, fusion, ranking) is hand-built, and the
BM25 side uses no model at all.

## The messy corpus

`data/corpus.jsonl` is built from real job-listing PDFs Sujal collected
during his job search (HIRINGS.pdf, Fresher_Jobs_India_Oct2026.pdf,
ML_GenAI_Jobs_India_Oct2026.pdf, Companies_Hiring_Junior_ML_AI_India.pdf,
TheUpskillBites_BTech_Engineering_Opportunities_02Nov2026.pdf). It is
genuinely messy: page footers chunked as documents, URL-only chunks,
entries split across chunks by page breaks, 20+ chunks truncated
mid-word, salary figures in five formats, tables flattened into lines.
`data/messiness_log.md` records what breaks and what was done about it.

No personal documents are in the corpus (no bank statements, IDs, or
certificates).

## Layout

```
src/
  corpus.py        # PDF -> text -> chunks -> corpus.jsonl
  tokenizer.py     # shared tokenizer
  bm25.py          # BM25 from scratch
  dense.py         # dense retrieval, numpy only
  fusion.py        # reciprocal rank fusion
  reranker.py      # cross-encoder reranking
  pipeline.py      # BM25 + dense -> fuse -> rerank
  evaluate.py      # Recall@k, MRR, nDCG
  evaluate_all.py  # BM25 vs dense vs fused vs hybrid
data/
  corpus.jsonl     # 121 chunks, built by corpus.py
  queries.json     # 25 labelled queries
  messiness_log.md # what breaks, and what was done about it
```

## Pipeline

BM25 top-20 and dense top-20 are fused with RRF (k=60), then the fused
top-20 is reranked with a cross-encoder and the top-5 returned. Results
on the 25 labelled queries (k=5): BM25 Recall@5 0.464, dense 0.359,
fused 0.464, hybrid 0.585. Fusion alone does not beat BM25; the gain
comes from the reranker. See `data/messiness_log.md` for the full
numbers and the failure modes.

## Run

```bash
pip install -r requirements.txt
python src/corpus.py          # build the corpus
python src/pipeline.py        # run the full pipeline on 4 demo queries
python src/evaluate_all.py    # evaluate all four systems
```

## Honesty rules (for the essay)

- Never claim a component is built until it runs and is measured.
- Numbers in the essay come from `evaluate.py` output, not from memory.
- Failure modes are features of the write-up, not things to hide —
  "survived messy data" means showing what broke and what fixed it.
