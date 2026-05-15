"""
retriever.py — Finds the most relevant chunks for a user query.

Location: rag_enterprise/retriever.py

What does the Retriever do?
  The Retriever is the "R" in RAG. It bridges the Embedder and the
  VectorStore to answer one question:

    "Given a user's question, which chunks from my document
     are most relevant to answer it?"

  It does this in two steps:
    1. Embed the query → convert question text into a vector
    2. Search the vector store → find the top-K closest chunk vectors

  The returned chunks are then passed to the LLM as context.

Usage:
  from retriever import Retriever
  retriever = Retriever()
  chunks = retriever.retrieve("What is the company leave policy?")
"""

from embedder import Embedder
from vector_store import VectorStore
from config import TOP_K_RESULTS


class Retriever:
    """
    Orchestrates query embedding and vector search to retrieve
    the most relevant document chunks for a given user question.

    Lifecycle:
      1. Instantiate — loads the embedder model and connects to ChromaDB
      2. retrieve() — call for each user question at query time
    """

    def __init__(self) -> None:
        """
        Initialise the Embedder and VectorStore.
        Both are loaded once and reused across multiple retrieve() calls.
        """
        print("[Retriever] Initialising...")
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        print("[Retriever] ✓ Ready")

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
    ) -> list[dict]:
        """
        Retrieve the top-K most relevant chunks for a user query.

        Steps:
          1. Validate the query is not empty
          2. Embed the query text into a 384-dim vector
          3. Search ChromaDB for the closest chunk vectors
          4. Filter out low-quality results (very high distance)
          5. Return enriched result dicts

        Input:
          query : natural language question from the user
          top_k : how many chunks to retrieve (default from config: 3)

        Output:
          [
            {
              "chunk_id"  : "chunk_4",
              "text"      : "Employees are entitled to 20 days...",
              "distance"  : 0.082,
              "word_count": 118,
              "rank"      : 1,       # 1 = most relevant
            },
            ...
          ]

        Returns [] if the vector store is empty or no good matches found.
        """
        # ── Step 1: Validate input ────────────────────────
        query = query.strip()
        if not query:
            raise ValueError("[Retriever] Query cannot be empty.")

        print(f"\n[Retriever] Query: \"{query}\"")

        # ── Step 2: Embed the query ───────────────────────
        print("[Retriever] Embedding query...")
        query_embedding = self.embedder.embed_query(query)

        # ── Step 3: Search the vector store ──────────────
        print(f"[Retriever] Searching for top-{top_k} chunks...")
        raw_results = self.vector_store.search(query_embedding, top_k=top_k)

        if not raw_results:
            print("[Retriever] ⚠ No results returned. "
                  "Is the vector store populated?")
            return []

        # ── Step 4: Filter very low-quality results ───────
        # Cosine distance range: 0.0 (identical) to 2.0 (opposite)
        # Anything above 1.0 is essentially unrelated — discard it
        DISTANCE_THRESHOLD = 1.0
        filtered = [r for r in raw_results if r["distance"] <= DISTANCE_THRESHOLD]

        if not filtered:
            print("[Retriever] ⚠ All results exceeded distance threshold "
                  f"({DISTANCE_THRESHOLD}). Query may be out of scope.")
            return []

        # ── Step 5: Add rank metadata and return ──────────
        ranked_results = []
        for rank, result in enumerate(filtered, start=1):
            ranked_results.append({**result, "rank": rank})

        # Print a summary for visibility
        print(f"[Retriever] ✓ Retrieved {len(ranked_results)} chunks:")
        for r in ranked_results:
            preview = r["text"][:70].replace("\n", " ")
            print(f"  #{r['rank']} [{r['chunk_id']}] "
                  f"distance={r['distance']} | {preview}...")

        return ranked_results

    def format_context(self, retrieved_chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a single context string for the LLM prompt.

        Each chunk is labelled with its rank so the LLM can reference it.
        Chunks are separated by a divider for clarity.

        Input:
          [{"rank": 1, "text": "...", ...}, {"rank": 2, ...}, ...]

        Output:
          "[Source 1]\nEmployees are entitled to 20 days...\n\n
           [Source 2]\nRemote work is permitted up to 3 days..."
        """
        if not retrieved_chunks:
            return "No relevant context found."

        sections = []
        for chunk in retrieved_chunks:
            sections.append(f"[Source {chunk['rank']}]\n{chunk['text'].strip()}")

        return "\n\n".join(sections)


# ─────────────────────────────────────────────
# Quick smoke test — run this file directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Retriever — Smoke Test")
    print("=" * 55)

    retriever = Retriever()

    # Check if vector store has data
    count = retriever.vector_store.count()
    if count == 0:
        print("\n⚠ Vector store is empty.")
        print("  Run the full ingestion pipeline first:")
        print("  → python rag_pipeline.py --ingest")
    else:
        print(f"\n  Vector store has {count} chunks. Running retrieval test...")

        test_query = "What are the main topics covered in this document?"
        results = retriever.retrieve(test_query)

        if results:
            context = retriever.format_context(results)
            print("\n── Formatted Context ─────────────────────────────")
            print(context[:500] + ("..." if len(context) > 500 else ""))
            print("\n✅ Retriever smoke test passed!")
        else:
            print("\n⚠ No results retrieved. Check your vector store.")