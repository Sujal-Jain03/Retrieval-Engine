# A Retrieval Engine Built From Scratch, Tested on Messy Data

Sujal Jain — B.Tech, Mathematics & Computing, Delhi Technological University (2026)

## What I built

I built a retrieval engine from scratch — BM25, dense retrieval,
reciprocal rank fusion, and cross-encoder reranking — and tested it on
the messiest corpus I could find: five job-listing PDFs I had collected
during my own job search, 121 chunks in total. Nothing in the pipeline
is a black box I didn't open: I implemented BM25 by hand in pure Python,
wired a pretrained bi-encoder for dense retrieval, fused the two
rankings with RRF, and reranked the fused top-20 with a cross-encoder.
To measure it honestly I hand-labelled 25 queries with 65 relevance
judgments before running a single evaluation, so the numbers below are
against labels I committed to in advance, not labels I adjusted to
flatter the system.

## The corpus was the hard part

The corpus was the hard part, and deliberately so. The PDFs were never
meant to be a dataset: 9 chunks are page footers, 6 are bare URLs, 7 job
entries are split across two chunks by a page break, and more than 20
chunks end mid-word. One PDF is mostly scanned images — only 1,476
characters of text came out of it. My assumptions about the data were
wrong in instructive ways: I expected Hindi text and found none; I
expected the phrase "data analyst" and it appears nowhere; the word
"stipend" appears only in chunks that say the stipend is not stated. I
logged every one of these defects instead of cleaning them away, because
the point of the project was to build something that survives real data,
and real data is like this. A retriever that only works on clean text is
a demo, not a system.

## How it works, mechanically

Each stage exists for a reason I can state. BM25 scores exact term
matches with term-frequency saturation, inverse document frequency, and
length normalisation — and I saw its failure mode directly: short footer
chunks score highly because length normalisation rewards brevity, so a
page footer can outrank a real job listing. Dense retrieval embeds query
and chunk separately with a 384-dimensional bi-encoder and ranks by
cosine similarity; it understands paraphrase ("entry level" for
"fresher") but cannot match exact tokens — it returns random chunks for
a job ID that BM25 finds instantly. The two scores live on different
scales, so they cannot be added; RRF fuses ranks instead, giving each
document 1/(60 + rank) from each retriever. Finally a cross-encoder
reads each (query, document) pair together, which is why it is more
accurate — and why it only reranks the fused top-20: scoring all 121
documents would be six times slower for little gain. The full pipeline
is BM25 top-20 plus dense top-20, fused, reranked, top-5 out. One bug
taught me more than the clean runs: I named my tokenizer module
tokenize.py, which collided with Python's own standard-library module of
the same name and broke the embedding library's imports. The fix was a
rename; the lesson was about import order and namespace collisions, the
kind of thing you only learn by hitting it.

## What I measured

Measured on 25 labelled queries at k=5, the four systems rank as
follows: BM25 — Recall@5 0.464, MRR 0.503, nDCG@5 0.555; dense — 0.359,
0.400, 0.442; fused (RRF, no rerank) — 0.464, 0.441, 0.518; hybrid
(fused + rerank) — 0.585, 0.660, 0.685. Two findings matter. First, BM25
beats dense outright, because this query set is keyword-heavy — company
names, job IDs, locations — and the labels reward exact matches. Second,
fusion alone does not beat BM25: it ties on recall and loses on MRR and
nDCG. The gain comes from the reranker: the hybrid beats the fused
system by 0.12 Recall@5. Concretely, the reranker lifts the true
stipend chunk — a Sony internship at Rs. 75,000/month — from fused #5
to #2, and turns four partial results into perfect ones. The ablation
exists precisely so I can say which component earned the improvement,
and the answer is the reranker, not the fusion.

## Where it fails, honestly

The failures are as informative as the wins. The reranker hurts one
query outright: on "machine learning internship stipend" it drops both
relevant chunks out of the top-5 (fused 0.50, hybrid 0.00), and its
scores there are all negative — the model is telling me it is not
confident, and I should listen to that signal instead of trusting the
ranking. Four queries score zero for every system, and three of them are
corpus gaps rather than retriever bugs: there are almost no genuine
work-from-home fresher jobs in the corpus, the word "laptop" never
appears, and there are no GenAI fresher listings at all. Retrieval
cannot find what is not there, and the honest response is to say so, not
to tune the retriever until it hallucinates an answer. The fourth, a
data-analyst internship query, has relevant chunks that no system ranks
in the top-5 — a genuine retrieval failure I have not yet fixed.

## What next

Next I would OCR the image-heavy PDF instead of accepting 1,476
characters, re-chunk on entry boundaries so no job is split by a footer,
and route queries by type — exact-token queries weighted toward BM25,
paraphrases toward dense — instead of fusing blindly. I would also grow
the labelled set: 25 queries is enough to rank four systems, not enough
to trust small differences between them. I built this because Auric's
work sits where retrieval meets reasoning, and I wanted to arrive with
something I had actually measured rather than something I had only read
about. The engine is small, the corpus is ugly, and the numbers are
honest — that is the point.
