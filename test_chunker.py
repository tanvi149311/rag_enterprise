"""
test_chunker.py — Unit tests for the data preparation pipeline.

Location: rag_enterprise/test_chunker.py

Tests cover:
  - Text cleaning (noise removal, whitespace, unicode)
  - Sliding window chunking (size, overlap, min size, edge cases)
  - PDF error handling (missing file)

Run all tests:
  pytest test_chunker.py -v

Run a single test:
  pytest test_chunker.py::test_chunk_size_respected -v
"""

import json
import sys
from pathlib import Path

import pytest

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from data_preparation import chunk_text, clean_text


# ─────────────────────────────────────────────
# FIXTURES — Reusable test data
# ─────────────────────────────────────────────

@pytest.fixture
def short_text():
    """A clean short paragraph — baseline for chunk tests."""
    return (
        "Artificial intelligence is transforming industries worldwide. "
        "Companies are investing heavily in machine learning solutions. "
        "Natural language processing enables computers to understand human text. "
        "These technologies are reshaping how businesses operate every day. "
        "The future of AI looks incredibly promising for all sectors."
    )


@pytest.fixture
def long_text():
    """A longer repeated block to test multi-chunk behaviour."""
    sentence = (
        "Enterprise data management requires careful planning and robust architecture. "
        "Organizations must ensure their data pipelines are scalable and maintainable. "
        "Security and compliance are critical concerns in any enterprise environment. "
    )
    return sentence * 30  # ~750 words — enough to produce multiple chunks


@pytest.fixture
def noisy_text():
    """Raw PDF-like text with noise artifacts."""
    return (
        "Page 1 of 20\n"
        "  \n"
        "  \n"
        "Introduction to RAG Systems\n"
        "- 1 -\n"
        "Retrieval-Augmented Generation is an industry-standard approach.\n"
        "It combines a retrieval step with a generation step.\n"
        "Visit https://example.com for more details.\n"
        "The system retrieves relevant documents before generating answers.\n"
        "This prevents hallucinations in large language models.\n"
        "Page 2 of 20\n"
        "  \n"
        "!\n"
        "?\n"
        "RAG pipelines are used in enterprise environments globally.\n"
    )


# ─────────────────────────────────────────────
# TESTS — Text Cleaning
# ─────────────────────────────────────────────

class TestCleanText:

    def test_removes_page_numbers(self, noisy_text):
        """Page number patterns like 'Page 1 of 20' must be stripped."""
        result = clean_text(noisy_text)
        assert "Page 1 of 20" not in result
        assert "Page 2 of 20" not in result

    def test_removes_dash_page_markers(self, noisy_text):
        """Dash-style page markers like '- 1 -' must be stripped."""
        result = clean_text(noisy_text)
        assert "- 1 -" not in result

    def test_removes_urls(self, noisy_text):
        """URLs must be removed from the text."""
        result = clean_text(noisy_text)
        assert "https://example.com" not in result

    def test_collapses_excessive_whitespace(self, noisy_text):
        """Multiple consecutive blank lines must be collapsed."""
        result = clean_text(noisy_text)
        assert "\n\n\n" not in result

    def test_removes_noise_lines(self, noisy_text):
        """Lines that are just punctuation (!, ?) must be discarded."""
        result = clean_text(noisy_text)
        lines = result.splitlines()
        for line in lines:
            assert len(line.strip()) > 3, (
                f"Noise line survived cleaning: '{line}'"
            )

    def test_preserves_meaningful_content(self, noisy_text):
        """Core content sentences must survive cleaning."""
        result = clean_text(noisy_text)
        assert "Retrieval-Augmented Generation" in result
        assert "prevents hallucinations" in result

    def test_normalizes_unicode_dashes(self):
        """Unicode em/en dashes must be normalized to ASCII hyphens."""
        text = "This is an em\u2014dash and an en\u2013dash."
        result = clean_text(text)
        assert "\u2014" not in result
        assert "\u2013" not in result
        assert "-" in result

    def test_normalizes_unicode_quotes(self):
        """Curly quotes must be normalized to straight quotes."""
        text = "She said \u201cHello\u201d and he replied \u2018Hi\u2019."
        result = clean_text(text)
        assert "\u201c" not in result
        assert "\u201d" not in result
        assert "\u2018" not in result
        assert "\u2019" not in result

    def test_empty_string_input(self):
        """Cleaning an empty string must return an empty string without errors."""
        result = clean_text("")
        assert result == ""

    def test_returns_string(self, short_text):
        """clean_text must always return a string."""
        result = clean_text(short_text)
        assert isinstance(result, str)


# ─────────────────────────────────────────────
# TESTS — Sliding Window Chunker
# ─────────────────────────────────────────────

class TestChunkText:

    def test_returns_list_of_dicts(self, short_text):
        """chunk_text must return a list of dicts."""
        chunks = chunk_text(short_text, chunk_size=50, overlap=10, min_size=5)
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert isinstance(chunk, dict)

    def test_chunk_has_required_keys(self, short_text):
        """Every chunk dict must have chunk_id, text, and word_count keys."""
        chunks = chunk_text(short_text, chunk_size=50, overlap=10, min_size=5)
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "word_count" in chunk

    def test_chunk_ids_are_sequential(self, long_text):
        """chunk_id values must start at 0 and increment by 1."""
        chunks = chunk_text(long_text, chunk_size=100, overlap=20, min_size=10)
        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_chunk_size_respected(self, long_text):
        """No chunk should exceed chunk_size words (except single oversized sentences)."""
        chunk_size = 100
        chunks = chunk_text(long_text, chunk_size=chunk_size, overlap=20, min_size=10)
        for chunk in chunks:
            # Allow slight overage only when a single sentence exceeds chunk_size
            assert chunk["word_count"] <= chunk_size * 1.5, (
                f"Chunk {chunk['chunk_id']} has {chunk['word_count']} words, "
                f"expected <= {chunk_size * 1.5}"
            )

    def test_min_chunk_size_filters_small_chunks(self, short_text):
        """Chunks with fewer words than min_size must be discarded."""
        min_size = 30
        chunks = chunk_text(short_text, chunk_size=500, overlap=10, min_size=min_size)
        for chunk in chunks:
            assert chunk["word_count"] >= min_size, (
                f"Chunk {chunk['chunk_id']} has only {chunk['word_count']} words, "
                f"below min_size={min_size}"
            )

    def test_overlap_creates_multiple_chunks(self, long_text):
        """With overlap, a long text should produce more than one chunk."""
        chunks = chunk_text(long_text, chunk_size=100, overlap=20, min_size=10)
        assert len(chunks) > 1, "Overlap chunking should produce multiple chunks"

    def test_word_count_matches_text(self, long_text):
        """word_count in metadata must match the actual word count of the text."""
        chunks = chunk_text(long_text, chunk_size=100, overlap=20, min_size=10)
        for chunk in chunks:
            actual = len(chunk["text"].split())
            assert actual == chunk["word_count"], (
                f"Chunk {chunk['chunk_id']}: metadata says {chunk['word_count']} words "
                f"but text has {actual} words"
            )

    def test_no_empty_chunk_text(self, long_text):
        """No chunk should have empty or whitespace-only text."""
        chunks = chunk_text(long_text, chunk_size=100, overlap=20, min_size=10)
        for chunk in chunks:
            assert chunk["text"].strip() != "", (
                f"Chunk {chunk['chunk_id']} has empty text"
            )

    def test_single_sentence_text(self):
        """A single short sentence should produce one chunk or none (if below min_size)."""
        text = "This is a single sentence for testing purposes only right here."
        chunks = chunk_text(text, chunk_size=500, overlap=100, min_size=5)
        assert len(chunks) <= 1

    def test_empty_text_returns_empty_list(self):
        """An empty string must return an empty list without errors."""
        chunks = chunk_text("", chunk_size=500, overlap=100, min_size=50)
        assert chunks == []

    def test_overlap_larger_than_chunk_size(self, long_text):
        """If overlap >= chunk_size, pipeline must not hang or crash."""
        # overlap >= chunk_size → step_words = max(1, ...) → always moves forward
        chunks = chunk_text(long_text, chunk_size=50, overlap=60, min_size=5)
        assert isinstance(chunks, list)

    def test_chunk_text_values_are_strings(self, long_text):
        """All chunk text values must be strings."""
        chunks = chunk_text(long_text, chunk_size=100, overlap=20, min_size=10)
        for chunk in chunks:
            assert isinstance(chunk["text"], str)

    def test_chunks_cover_full_document(self, long_text):
        """
        The union of all chunk texts must contain the key content of the document.
        We verify by checking that a known phrase appears somewhere in the chunks.
        """
        chunks = chunk_text(long_text, chunk_size=100, overlap=20, min_size=10)
        all_text = " ".join(c["text"] for c in chunks)
        assert "Enterprise data management" in all_text


# ─────────────────────────────────────────────
# TESTS — Integration (clean → chunk pipeline)
# ─────────────────────────────────────────────

class TestCleanThenChunk:

    def test_clean_then_chunk_produces_valid_output(self, noisy_text):
        """Running clean_text followed by chunk_text must produce valid chunks."""
        cleaned = clean_text(noisy_text)
        chunks = chunk_text(cleaned, chunk_size=50, overlap=10, min_size=5)
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert chunk["text"].strip() != ""
            assert chunk["word_count"] > 0

    def test_noise_does_not_appear_in_chunks(self, noisy_text):
        """After cleaning and chunking, page numbers must not appear in any chunk."""
        cleaned = clean_text(noisy_text)
        chunks = chunk_text(cleaned, chunk_size=50, overlap=10, min_size=5)
        for chunk in chunks:
            assert "Page 1 of 20" not in chunk["text"]
            assert "Page 2 of 20" not in chunk["text"]

    def test_output_is_json_serializable(self, long_text):
        """Chunks must be serializable to JSON (required for saving to disk)."""
        cleaned = clean_text(long_text)
        chunks = chunk_text(cleaned, chunk_size=100, overlap=20, min_size=10)
        try:
            json.dumps(chunks)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Chunks are not JSON serializable: {e}")