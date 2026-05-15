"""
embedder.py — Converts text chunks into vector embeddings.

Location: rag_enterprise/embedder.py

What is an embedding?
  A vector embedding is a list of numbers (e.g. 384 floats) that captures
  the *semantic meaning* of a piece of text. Two sentences with similar
  meanings will have vectors that are mathematically close to each other.
  This is what powers semantic search in the RAG pipeline.

Model used:
  all-MiniLM-L6-v2 (via sentence-transformers)
  - Runs 100% locally — no API key needed
  - Downloads once (~80MB) on first run
  - Produces 384-dimensional embeddings
  - Fast on CPU, high quality for retrieval tasks

Usage:
  from embedder import Embedder
  embedder = Embedder()
  embeddings = embedder.embed_chunks(chunks)
"""

from typing import Any

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class Embedder:
    """
    Wraps the sentence-transformers model to convert text into embeddings.

    Lifecycle:
      1. Instantiate once (loads model into memory)
      2. Call embed_chunks() or embed_query() as many times as needed
      3. The same model instance is reused — no repeated loading
    """

    def __init__(self) -> None:
        """
        Load the embedding model into memory.
        On first run this downloads the model weights (~80MB).
        Subsequent runs load from local cache instantly.
        """
        print(f"[Embedder] Loading model: {EMBEDDING_MODEL}")
        print("           (downloading ~80MB on first run — cached after that)")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] ✓ Model ready — embedding dimension: {self.embedding_dim}")

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Convert a list of chunk dicts into enriched dicts with embeddings.

        Input:
          [{"chunk_id": 0, "text": "...", "word_count": 120}, ...]

        Output:
          [{"chunk_id": 0, "text": "...", "word_count": 120,
            "embedding": [0.021, -0.143, ...]}, ...]

        The embedding field is a list of 384 floats representing the
        semantic meaning of the chunk text.
        """
        if not chunks:
            print("[Embedder] Warning: No chunks provided to embed.")
            return []

        print(f"[Embedder] Embedding {len(chunks)} chunks...")

        # Extract just the text for batch embedding (much faster than one-by-one)
        texts = [chunk["text"] for chunk in chunks]

        # encode() returns a numpy array of shape (num_chunks, embedding_dim)
        # show_progress_bar gives live feedback for large document sets
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,          # Process 32 chunks at a time
            convert_to_numpy=True,  # Returns numpy array for easy slicing
        )

        # Attach each embedding back to its chunk dict
        enriched_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            enriched_chunks.append({
                **chunk,                            # Keep all existing fields
                "embedding": embedding.tolist(),    # Convert numpy → plain list
            })

        print(f"[Embedder] ✓ Embedded {len(enriched_chunks)} chunks "
              f"(dim={self.embedding_dim})")
        return enriched_chunks

    def embed_query(self, query: str) -> list[float]:
        """
        Convert a single user query string into an embedding vector.

        This is called at query time (not during ingestion).
        The query embedding is compared against stored chunk embeddings
        to find the most semantically similar chunks.

        Input:  "What is the company's leave policy?"
        Output: [0.021, -0.143, 0.087, ...]  (384 floats)
        """
        if not query.strip():
            raise ValueError("[Embedder] Query string cannot be empty.")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of embeddings produced by this model."""
        return self.embedding_dim


# ─────────────────────────────────────────────
# Quick smoke test — run this file directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Embedder — Smoke Test")
    print("=" * 55)

    embedder = Embedder()

    # Test with sample chunks
    sample_chunks = [
        {"chunk_id": 0, "text": "The company offers 20 days of paid leave annually.", "word_count": 9},
        {"chunk_id": 1, "text": "Remote work is allowed up to 3 days per week.", "word_count": 10},
        {"chunk_id": 2, "text": "All employees must complete mandatory safety training.", "word_count": 8},
    ]

    enriched = embedder.embed_chunks(sample_chunks)

    print("\n── Results ───────────────────────────────────────")
    for chunk in enriched:
        preview = chunk["embedding"][:5]  # Show first 5 floats only
        print(f"  Chunk {chunk['chunk_id']}: {chunk['text'][:50]}...")
        print(f"    Embedding preview: {[round(v, 4) for v in preview]}...")

    # Test query embedding
    query_vec = embedder.embed_query("How many leave days do employees get?")
    print(f"\n  Query embedding preview: {[round(v, 4) for v in query_vec[:5]]}...")
    print(f"  Embedding dimension: {len(query_vec)}")
    print("\n✅ Embedder smoke test passed!")