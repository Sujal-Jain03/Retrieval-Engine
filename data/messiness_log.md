# Messiness Log

What breaks on real data, and what was done about it. This log is essay
material — "survived messy data" is proved here, not claimed.

## 2026-11-13 — Corpus built (Day 1)
- 121 chunks, avg ~306 chars. Sources: Fresher_Jobs (38), Companies_Hiring (35),
  ML_GenAI_Jobs (31), UpskillBites (16), HIRINGS (1).
- HIRINGS.pdf is image-heavy: 1 page, only 1,476 chars extractable via
  pdftotext — most of its content is screenshots, so text extraction
  alone misses it. OCR (tesseract) would be needed to include it properly.
  Lesson logged: "PDF" does not mean "text".
- Chunks carry real mess: repeated listings across PDFs, truncated lines,
  salary in multiple formats, flattened tables, mixed Hindi-English.

## 2026-11-14 — Corpus inspected (Phase 1)
- Duplicates across PDFs: Siemens Energy AI/ML Engineer 298438 appears
  twice (Fresher_Jobs_India_Oct2026-6 and ML_GenAI_Jobs_India_Oct2026-2)
  with different experience text (1+ years OR strong projects vs 0-1
  years) and different pay (Not disclosed vs Rs. 8-12 LPA estimated).
  Same for GlobalLogic AI Engineer IRC305438
  (Fresher_Jobs_India_Oct2026-11 and ML_GenAI_Jobs_India_Oct2026-13).
  A retriever returns both copies as if they were two jobs.
- Split entries: 7 job entries are split across two chunks by a page
  footer in between — AMETEK (Fresher_Jobs_India_Oct2026-7 + -9),
  Cisco Trainee (-14 + -16), TCS NQT (-21 + -23), HPE (-28 + -30),
  S&P Global (-35 + -37), Analog Devices
  (ML_GenAI_Jobs_India_Oct2026-6 + -8), Shyena Tech Yarns (-20 + -22).
  Returning only one half gives an incomplete job.
- Noise chunks: 9 page-footer chunks ("Page X/6 ...", e.g.
  Fresher_Jobs_India_Oct2026-8) and 6 URL-only chunks (e.g.
  Companies_Hiring_Junior_ML_AI_India-5) carry no job information.
  Section headers (e.g. Fresher_Jobs_India_Oct2026-3 "Section A -
  Verified live") also carry none.
- Truncation: 20+ chunks in Companies_Hiring_Junior_ML_AI_India are
  cut mid-word ("Fit: Runs Python/M", "Fit: Heav", "Fit: Analy junior
  ma"). Words also break across lines inside URLs ("Enginee ring-UHR"
  in Fresher_Jobs_India_Oct2026-12, "locat ionId" in -32).
- Salary in many formats: "Rs. 4-8 LPA", "Rs. 75,000/month",
  "Rs. 3-25 LPA", "Rs. 4.5 LPA", "Up to Rs. 5 LPA",
  "Rs. 9-15 LPA + ESOPs", and "Not disclosed" most often. Numbers are
  text, not values — a query for one figure cannot match another.
- HIRINGS-0 is the opposite problem: ~20 job titles flattened into a
  single 1,200-char chunk with no structure. It matches almost any job
  query (keyword soup).
- Closed roles look live: Section B (Fresher_Jobs_India_Oct2026-24
  onwards) is all closed/unverified — HPE "job has been filled", Ford
  "404 (posting removed)", United "CLOSED as of Oct 6" — but the chunks
  read like normal listings. Nothing marks them dead except the word
  CLOSED, which a keyword retriever does not understand.
- Contradictions inside the text: Cyncly posting "header says Kochi
  while description says Bengaluru" (ML_GenAI_Jobs_India_Oct2026-4);
  Johnson Controls "conflicting IDs circulate" (-15); Skyhigh "recent
  college graduate but also demands 2+ years"
  (Fresher_Jobs_India_Oct2026-33).
- Correction to the 2026-11-13 entry: no Hindi/Devanagari text was
  found anywhere in the 121 chunks (checked programmatically). The
  mess here is structural — splits, truncation, duplicates — not
  multilingual.
## 2026-11-14 — BM25 implemented (Phase 2)
- BM25 from scratch in src/bm25.py (k1=1.5, b=0.75): inverted index,
  121 docs, avgdl 45.7 tokens, 1243 unique terms.
- "jobs in accenture for freshers" returns the Accenture chunk first
  (Fresher_Jobs_India_Oct2026-10, score 7.89) — rare terms win because
  IDF("accenture") = 4.40 vs IDF("engineer") = 0.94.
- "machine learning internship stipend" top-3 are all ML Engineer
  jobs (ML_GenAI_Jobs_India_Oct2026-2, -19, -10) — none mentions an
  internship or a stipend. Keyword matching cannot tell that "machine
  learning" and "internship stipend" are different intents in one query.
- "paid internship stipend per month" top result is Skyhigh
  (Fresher_Jobs_India_Oct2026-33), which matched "3-month internship"
  — not a stipend chunk. The word "stipend" only matches chunks saying
  stipend is NOT stated.
- Length normalisation inflates tiny docs: the page-footer chunk
  Fresher_Jobs_India_Oct2026-8 ("Page 1/6 ...", 11 tokens) ranks #3
  for "fresher data analyst jobs in Hyderabad" with score 5.99.
- No stemming: "fresh graduates" tokenises to "fresh", so the query
  term "fresher" does not match it.
## 2026-11-14 — Dense retrieval implemented (Phase 3)
- DenseRetriever in src/dense.py: all-MiniLM-L6-v2 embeddings
  (384-dim), L2-normalized, single (121, 384) numpy matrix; search is
  one matrix-vector product, no vector DB.
- BM25 vs dense (top-1 per query):
  - "fresher data analyst hyderabad": BM25 ML_GenAI_Jobs_India_Oct2026-12,
    dense Companies_Hiring_Junior_ML_AI_India-2 (0.614); dense #2
    Companies_Hiring_Junior_ML_AI_India-7 (0.527), #3
    Companies_Hiring_Junior_ML_AI_India-33 (0.517). Labelled
    ML_GenAI_Jobs_India_Oct2026-24 is not in the dense top-3.
  - "machine learning internship stipend": BM25 ML_GenAI_Jobs_India_Oct2026-2,
    dense HIRINGS-0 (labelled relevant — dense wins).
  - "jobs in accenture for freshers": both Fresher_Jobs_India_Oct2026-10;
    dense #2/#3 are page-footer chunks (-8, -15).
  - "paid internship stipend per month": BM25 Fresher_Jobs_India_Oct2026-33,
    dense Fresher_Jobs_India_Oct2026-32; the true stipend chunk
    (Fresher_Jobs_India_Oct2026-20, Sony ~Rs. 75,000/month) is BM25 #2
    (5.98) but not in the dense top-3; dense top score only 0.415.
  - "fresher data analyst jobs in Hyderabad": BM25 ML_GenAI_Jobs_India_Oct2026-12,
    dense ML_GenAI_Jobs_India_Oct2026-23 (dense #2 is the labelled -24).
  - "entry level jobs for new graduates": BM25 Fresher_Jobs_India_Oct2026-19,
    dense Fresher_Jobs_India_Oct2026-14; both sensible.
  - "monthly pay for interns": BM25 TheUpskillBites_BTech_Engineering_Opportunities_02Nov2026-5,
    dense TheUpskillBites_BTech_Engineering_Opportunities_02Nov2026-8; both weak.
  - "jobs you can do from home": BM25 Fresher_Jobs_India_Oct2026-24 (section header),
    dense Companies_Hiring_Junior_ML_AI_India-12 (URL-only chunk); both fail.
  - "26009713": BM25 Fresher_Jobs_India_Oct2026-4 (only result) — exact ID match;
    dense returns unrelated chunks at ~0.19. BM25 wins decisively on exact tokens.
- Pattern: dense wins on meaning (internship/stipend intent), BM25 wins on
  exact tokens (job IDs, company names). Both fail on vague queries and on
  the stipend question. Both surface junk: BM25 inflates short footer chunks
  via length normalisation; dense ranks short generic texts high.

## 2026-11-14 — Fusion + reranking implemented (Phase 4)
- RRF fusion in src/fusion.py (k=60): fuses BM25 + dense rankings by
  rank, not score. Cross-encoder reranker in src/reranker.py
  (ms-marco-MiniLM-L-6-v2): re-scores the fused top-20, returns top-5.
  Pipeline in src/pipeline.py wires all four stages.
- "fresher data analyst hyderabad": fused #1 Companies_Hiring_Junior_ML_AI_India-2
  (0.0323, BM25 #3 + dense #1); reranked #1 Companies_Hiring_Junior_ML_AI_India-8
  (1.117), #2 ML_GenAI_Jobs_India_Oct2026-23 (0.495). Reranking dropped
  the header chunk -2 out of the top-5, but also dropped the labelled
  ML_GenAI_Jobs_India_Oct2026-24.
- "machine learning internship stipend": fused top-5 is a tight cluster
  (0.0307-0.0315) mixing both retrievers; reranked top-3 are all
  internship chunks (-5, -9, -8). All rerank scores are negative —
  the cross-encoder is not confident about any of them.
- "paid internship stipend per month": reranker puts the Amazon intern
  chunk TheUpskillBites_BTech_Engineering_Opportunities_02Nov2026-5 at
  #1 (0.140) and the true stipend chunk Fresher_Jobs_India_Oct2026-20
  (Sony, Rs. 75,000/month) at #2 (-0.257), up from fused #5. The top-2
  are both genuine internship listings.
- "26009713": fused #1 Fresher_Jobs_India_Oct2026-4 (0.0299, BM25 #1 +
  dense #14); reranked #1 the same chunk at +5.909 vs -5.971 for #2 —
  a 12-point gap. Exact-ID matches are where the cross-encoder is most
  confident.
- Cross-encoder scores are logits, not 0-1: negative scores are normal;
  only the order matters. Reranking runs on the top-20 only — scoring
  all 121 docs would be ~6x slower for little gain.
- Fix: renamed src/tokenize.py to src/tokenizer.py — the old name
  collided with Python's stdlib `tokenize` module and broke
  huggingface_hub imports (and would have broken the pipeline).



## 2026-11-14 — Evaluation (Phase 5): BM25 vs dense vs fused vs hybrid
- 25 labelled queries, 65 relevant labels. Metrics at k=5.
- Results: BM25 Recall@5 0.464 / MRR 0.503 / nDCG@5 0.555;
  Dense 0.359 / 0.400 / 0.442; Fused (RRF, no rerank) 0.464 / 0.441 /
  0.518; Hybrid (fused + cross-encoder rerank) 0.585 / 0.660 / 0.685.
- BM25 beats dense: this query set is keyword-heavy (company names,
  job IDs, locations, "2026 batch") and the labels reward exact
  matches; embeddings struggle with exact tokens.
- Fusion alone does NOT beat BM25: Fused ties BM25 on Recall@5 (0.464)
  but is below on MRR (0.441 vs 0.503) and nDCG@5 (0.518 vs 0.555).
  RRF dilutes BM25's exact-match wins with dense's misses.
- The reranker is where the gain comes from: Hybrid 0.585 vs Fused
  0.464 (+0.12 Recall@5). Reranking rescues the fused ranking —
  e.g. "paid internship stipend per month" goes 0.50 (fused) to 1.00
  (hybrid); "software engineer jobs chennai fresher" 0.67 to 1.00;
  "support engineer jobs noida fresher" 0.00 to 0.67.
- Reranker also regresses queries: "machine learning internship
  stipend" goes 0.50 (fused) to 0.00 (hybrid) — the cross-encoder's
  scores there are all negative (low confidence) and it drops both
  relevant chunks out of the top-5. "software engineer fresher
  Bangalore 2026 batch" goes 0.43 to 0.29.
- Four queries score 0.00 for every system: "work from home jobs
  for freshers", "fresher jobs work from home laptop", "genai engineer
  jobs india fresher", "data analyst internship work from home
  stipend". The first three are corpus gaps — the corpus has almost
  no true WFH fresher jobs, "laptop" never appears, and there are no
  GenAI fresher listings. Retrieval cannot find what is not there.
- Four queries are perfect (1.00) for every system: "software
  developer internship paid india", "qa engineer fresher jobs",
  "data engineer fresher jobs sql python", "jobs in accenture for
  freshers".
