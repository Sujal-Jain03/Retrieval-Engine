"""Build the corpus: job-listing PDFs -> chunked JSONL.

Uses pdftotext (poppler) — no PDF library. Chunking is paragraph-based
with a max length, so flattened tables and truncated lines survive as-is.
Run: python src/corpus.py -> data/corpus.jsonl (id, source, text).
Job listings only — no personal documents.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCES = [
    ROOT / "data" / "pdfs" / "HIRINGS.pdf",
    ROOT / "data" / "pdfs" / "Fresher_Jobs_India_Oct2026.pdf",
    ROOT / "data" / "pdfs" / "ML_GenAI_Jobs_India_Oct2026.pdf",
    ROOT / "data" / "pdfs" / "Companies_Hiring_Junior_ML_AI_India.pdf",
    ROOT / "data" / "pdfs" / "TheUpskillBites_BTech_Engineering_Opportunities_02Nov2026.pdf",
]

MAX_CHARS = 1200  # chunk cap; long paragraphs are split on sentence ends


def pdf_to_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def chunk_text(text: str) -> list[str]:
    """Split on blank lines, then hard-split overlong paragraphs."""
    chunks: list[str] = []
    for para in text.split("\n\n"):
        para = " ".join(para.split())
        if len(para) < 40:
            continue  # page numbers, headers, stray fragments
        while len(para) > MAX_CHARS:
            cut = para.rfind(". ", 0, MAX_CHARS)
            cut = cut + 1 if cut > 200 else MAX_CHARS
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            chunks.append(para)
    return chunks


def main() -> None:
    out_path = ROOT / "data" / "corpus.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Build in memory first: never overwrite a good corpus.jsonl with an empty one.
    rows: list[dict] = []
    for src in SOURCES:
        if not src.exists():
            print(f"SKIP (missing): {src}")
            continue
        chunks = chunk_text(pdf_to_text(src))
        for i, chunk in enumerate(chunks):
            rows.append({
                "id": f"{src.stem}-{i}",
                "source": src.name,
                "text": chunk,
            })
        print(f"{src.name}: {len(chunks)} chunks")
    if not rows:
        print("ERROR: 0 chunks built — corpus.jsonl left untouched.")
        print("Check that data/pdfs/ has the 5 PDFs and pdftotext is installed.")
        return
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Total: {len(rows)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
