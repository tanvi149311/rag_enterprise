"""
data_preparation.py — Part 1 of the RAG pipeline.

Does three things in sequence:
  1. Load raw text from a PDF file
  2. Clean the text using NLP techniques
  3. Split into overlapping chunks using a sliding window strategy

Run directly:
  python data_preparation.py
"""

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_FILE,
    MIN_CHUNK_SIZE,
    PDF_PATH,
)

# Download NLTK data on first run (silent after that)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ─────────────────────────────────────────────
# STEP 1 — PDF LOADER
# ─────────────────────────────────────────────

def load_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file page by page.
    Returns a single concatenated string of the full document.
    """
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found at: {pdf_path}")
        print("  → Please place your PDF at data/document.pdf and try again.")
        sys.exit(1)

    print(f"[1/3] Loading PDF: {pdf_path.name}")

    doc = fitz.open(str(pdf_path))
    pages_text = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")  # Extract plain text from page
        if text.strip():              # Skip blank pages
            pages_text.append(text)

    doc.close()

    full_text = "\n".join(pages_text)
    print(f"      ✓ Extracted text from {len(pages_text)} pages "
          f"({len(full_text):,} characters)")
    return full_text


# ─────────────────────────────────────────────
# STEP 2 — TEXT CLEANER
# ─────────────────────────────────────────────

def clean_text(raw_text: str) -> str:
    """
    Clean raw PDF text using NLP techniques:
      - Remove page numbers and headers/footers patterns
      - Collapse excessive whitespace and newlines
      - Remove special characters that add no meaning
      - Normalize unicode punctuation
      - Strip leading/trailing whitespace
    """
    print("[2/3] Cleaning text...")

    text = raw_text

    # Remove common PDF artifacts: page numbers like "Page 1 of 20", "- 1 -"
    text = re.sub(r"[-–]\s*\d+\s*[-–]", " ", text)
    text = re.sub(r"[Pp]age\s+\d+\s+(of\s+\d+)?", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Normalize unicode dashes, quotes, and ellipsis to ASCII equivalents
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2026", "...")

    # Remove non-printable / control characters (keep newlines)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ consecutive newlines into 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip lines that are just punctuation or single characters (noise)
    lines = text.splitlines()
    lines = [line for line in lines if len(line.strip()) > 3]
    text = "\n".join(lines)

    text = text.strip()
    print(f"      ✓ Cleaned text: {len(text):,} characters remaining")
    return text


# ─────────────────────────────────────────────
# STEP 3 — SLIDING WINDOW CHUNKER
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int, min_size: int) -> list[dict]:
    """
    Split text into overlapping chunks using a sliding window strategy.

    Strategy:
      - Tokenize the full text into sentences (respects sentence boundaries)
      - Accumulate sentences into a chunk until CHUNK_SIZE words is reached
      - Slide the window forward by (CHUNK_SIZE - CHUNK_OVERLAP) words
      - This ensures no sentence is awkwardly cut mid-way

    Returns:
      List of dicts: [{"chunk_id": int, "text": str, "word_count": int}]
    """
    print("[3/3] Chunking text with sliding window strategy...")

    # Tokenize into sentences for clean boundaries
    sentences = sent_tokenize(text)
    words_per_sentence = [len(s.split()) for s in sentences]

    chunks = []
    chunk_id = 0
    i = 0  # sentence index

    while i < len(sentences):
        chunk_sentences = []
        word_count = 0

        # Fill the chunk up to CHUNK_SIZE words
        j = i
        while j < len(sentences) and word_count + words_per_sentence[j] <= chunk_size:
            chunk_sentences.append(sentences[j])
            word_count += words_per_sentence[j]
            j += 1

        # If a single sentence exceeds chunk_size, include it anyway (don't skip)
        if not chunk_sentences and j < len(sentences):
            chunk_sentences.append(sentences[j])
            word_count = words_per_sentence[j]
            j += 1

        chunk_text_str = " ".join(chunk_sentences).strip()

        # Only keep chunks that meet the minimum size threshold
        if len(chunk_text_str.split()) >= min_size:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text_str,
                "word_count": word_count,
            })
            chunk_id += 1

        # Slide the window: step forward by (chunk_size - overlap) words
        step_words = max(1, chunk_size - overlap)
        stepped = 0
        while i < len(sentences) and stepped < step_words:
            stepped += words_per_sentence[i]
            i += 1

    print(f"      ✓ Created {len(chunks)} chunks "
          f"(avg {sum(c['word_count'] for c in chunks) // max(len(chunks), 1)} words/chunk)")
    return chunks


# ─────────────────────────────────────────────
# SAVE CHUNKS TO DISK
# ─────────────────────────────────────────────

def save_chunks(chunks: list[dict], output_path: Path) -> None:
    """Save chunks as a JSON file for use by the embedding pipeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(chunks)} chunks → {output_path}")


# ─────────────────────────────────────────────
# MAIN — Run the full Part 1 pipeline
# ─────────────────────────────────────────────

def run_data_preparation() -> list[dict]:
    """Execute the full data preparation pipeline and return chunks."""
    print("=" * 55)
    print("  RAG Pipeline — Part 1: Data Preparation")
    print("=" * 55)

    raw_text = load_pdf(PDF_PATH)
    clean = clean_text(raw_text)
    chunks = chunk_text(clean, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE)
    save_chunks(chunks, CHUNKS_FILE)

    # Preview first chunk
    if chunks:
        print("\n── Preview: First chunk ──────────────────────────")
        preview = chunks[0]["text"][:300]
        print(f"{preview}{'...' if len(chunks[0]['text']) > 300 else ''}")
        print(f"(word count: {chunks[0]['word_count']})")
        print("──────────────────────────────────────────────────")

    return chunks


if __name__ == "__main__":
    run_data_preparation()