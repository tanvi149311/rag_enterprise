"""
vector_store.py — ChromaDB wrapper for storing and searching embeddings.

Location: rag_enterprise/vector_store.py

What is a Vector Store?
  A vector store is a specialised database optimised for storing and
  searching high-dimensional vectors (embeddings). Unlike a regular
  database that matches exact keywords, a vector store finds the
  *semantically closest* vectors to a query vector using mathematical
  distance functions (cosine similarity).

Why ChromaDB?
  - Runs 100% locally — no server, no cloud, no setup
  - Persists data to disk (chroma_db/ folder)
  - Simple Python API
  - Fast enough for thousands of enterprise document chunks

Usage:
  from vector_store import VectorStore
  vs = VectorStore()
  vs.add_chunks(enriched_chunks)        # Store embeddings
  results = vs.search(query_embedding)  # Find top-K similar chunks
"""

import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    TOP_K_RESULTS,
)


class VectorStore:
    """
    Manages a ChromaDB collection for storing and retrieving chunk embeddings.

    Lifecycle:
      1. Instantiate — connects to (or creates) the local ChromaDB database
      2. add_chunks() — store embedded chunks during ingestion
      3. search() — find the top-K closest chunks at query time
      4. clear() — wipe and reset the collection (useful for re-ingestion)
    """

    def __init__(self) -> None:
        """
        Connect to the local ChromaDB instance.
        Creates the chroma_db/ directory and collection if they don't exist.
        If they already exist, connects to the existing data (no data loss).
        """
        print(f"[VectorStore] Connecting to ChromaDB at: {CHROMA_DIR}")

        # PersistentClient saves data to disk automatically
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),  # Disable telemetry
        )

        # get_or_create_collection: safe to call repeatedly
        # cosine similarity is best for semantic text search
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        count = self.collection.count()
        print(f"[VectorStore] ✓ Collection '{CHROMA_COLLECTION_NAME}' ready "
              f"({count} chunks already stored)")

    def add_chunks(self, enriched_chunks: list[dict]) -> None:
        """
        Store embedded chunks into ChromaDB.

        ChromaDB requires three parallel lists:
          - ids        : unique string ID per chunk  (e.g. "chunk_0")
          - embeddings : the 384-float vector
          - documents  : the raw text (stored for retrieval)
          - metadatas  : extra fields (word_count, chunk_id) for filtering

        Skips chunks that are already stored (safe for re-runs).

        Input:
          [{"chunk_id": 0, "text": "...", "word_count": 120,
            "embedding": [0.021, ...]}, ...]
        """
        if not enriched_chunks:
            print("[VectorStore] Warning: No chunks to add.")
            return

        # Build the four parallel lists ChromaDB expects
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        existing_ids = set(self.collection.get()["ids"])  # Already stored IDs

        skipped = 0
        for chunk in enriched_chunks:
            chunk_id_str = f"chunk_{chunk['chunk_id']}"

            if chunk_id_str in existing_ids:
                skipped += 1
                continue  # Don't duplicate

            ids.append(chunk_id_str)
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])
            metadatas.append({
                "chunk_id": chunk["chunk_id"],
                "word_count": chunk["word_count"],
            })

        if not ids:
            print(f"[VectorStore] All {skipped} chunks already stored. Skipping.")
            return

        # Upsert in batches of 500 to avoid memory issues with large docs
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

        total_stored = self.collection.count()
        print(f"[VectorStore] ✓ Added {len(ids)} chunks "
              f"(skipped {skipped} duplicates, "
              f"total in DB: {total_stored})")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K_RESULTS,
    ) -> list[dict]:
        """
        Find the top-K most semantically similar chunks to a query embedding.

        ChromaDB compares the query vector against every stored chunk vector
        using cosine similarity and returns the closest matches.

        Input:
          query_embedding : 384-float list from Embedder.embed_query()
          top_k           : number of results to return (default: 3)

        Output:
          [
            {
              "chunk_id"    : "chunk_4",
              "text"        : "Employees are entitled to 20 days...",
              "distance"    : 0.082,   # lower = more similar (cosine)
              "word_count"  : 118,
            },
            ...
          ]
        """
        if self.collection.count() == 0:
            print("[VectorStore] Warning: Collection is empty. "
                  "Run ingestion first.")
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],  # Wrap in list (batch API)
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Unpack ChromaDB's nested response format into clean dicts
        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "chunk_id"  : results["ids"][0][i],
                "text"      : results["documents"][0][i],
                "distance"  : round(results["distances"][0][i], 6),
                "word_count": results["metadatas"][0][i].get("word_count", 0),
            })

        return retrieved

    def count(self) -> int:
        """Return the total number of chunks currently stored in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """
        Delete and recreate the collection — wipes all stored chunks.
        Use this when re-ingesting a new or updated document.
        """
        print(f"[VectorStore] Clearing collection '{CHROMA_COLLECTION_NAME}'...")
        self.client.delete_collection(CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("[VectorStore] ✓ Collection cleared and recreated.")


# ─────────────────────────────────────────────
# Quick smoke test — run this file directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=" * 55)
    print("  VectorStore — Smoke Test")
    print("=" * 55)

    vs = VectorStore()
    vs.clear()  # Start fresh for the test

    # Simulate enriched chunks with random embeddings (dim=384)
    sample_chunks = [
        {
            "chunk_id"  : 0,
            "text"      : "Employees are entitled to 20 days of paid annual leave.",
            "word_count": 10,
            "embedding" : [random.uniform(-1, 1) for _ in range(384)],
        },
        {
            "chunk_id"  : 1,
            "text"      : "Remote work is permitted up to 3 days per week.",
            "word_count": 9,
            "embedding" : [random.uniform(-1, 1) for _ in range(384)],
        },
        {
            "chunk_id"  : 2,
            "text"      : "All staff must complete annual compliance training.",
            "word_count": 8,
            "embedding" : [random.uniform(-1, 1) for _ in range(384)],
        },
    ]

    vs.add_chunks(sample_chunks)
    print(f"\n  Total stored: {vs.count()} chunks")

    # Search with a random query vector
    query_vec = [random.uniform(-1, 1) for _ in range(384)]
    results = vs.search(query_vec, top_k=2)

    print("\n── Search Results ────────────────────────────────")
    for r in results:
        print(f"  [{r['chunk_id']}] distance={r['distance']} | {r['text'][:60]}...")

    vs.clear()  # Clean up after test
    print("\n✅ VectorStore smoke test passed!")