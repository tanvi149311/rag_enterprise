"""
rag_pipeline.py — Orchestrates the full end-to-end RAG pipeline.

Location: rag_enterprise/rag_pipeline.py

This is the brain of the project. It wires together every component:

  INGESTION (run once per document):
    PDF → clean text → chunks → embeddings → ChromaDB

  QUERYING (run for every user question):
    Question → embed → retrieve top-3 chunks → build prompt → LLM → answer

Two LLM providers are supported (toggled via .env):
  - Anthropic Claude  (LLM_PROVIDER=anthropic)
  - OpenAI GPT-4o     (LLM_PROVIDER=openai)

Usage:
  from rag_pipeline import RAGPipeline
  pipeline = RAGPipeline()
  pipeline.ingest()                          # Run once
  answer = pipeline.query("Your question")   # Run as many times as needed
"""

import json

import anthropic
import openai

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_MODEL,
    CHUNKS_FILE,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    PROMPT_TEMPLATE,
)
from data_preparation import run_data_preparation
from embedder import Embedder
from retriever import Retriever
from vector_store import VectorStore


class RAGPipeline:
    """
    End-to-end RAG pipeline that handles both ingestion and querying.

    Ingestion  — processes your PDF into a searchable vector database.
    Querying   — answers questions using retrieved context + an LLM.
    """

    def __init__(self) -> None:
        """
        Initialise all pipeline components.
        LLM clients are created here but API calls only happen during query().
        """
        print("=" * 55)
        print("  RAG Pipeline — Initialising")
        print("=" * 55)

        self.embedder     = Embedder()
        self.vector_store = VectorStore()
        self.retriever    = Retriever.__new__(Retriever)

        # Reuse already-loaded embedder and vector_store in retriever
        # (avoids loading the model twice)
        self.retriever.embedder     = self.embedder
        self.retriever.vector_store = self.vector_store

        # Set up LLM client based on provider selection
        self._setup_llm_client()

        print(f"\n[Pipeline] ✓ Ready — LLM provider: {LLM_PROVIDER.upper()}")

    def _setup_llm_client(self) -> None:
        """
        Initialise the correct LLM API client based on LLM_PROVIDER in .env.
        Raises a clear error if the required API key is missing.
        """
        if LLM_PROVIDER == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise EnvironmentError(
                    "[Pipeline] ANTHROPIC_API_KEY is not set in your .env file.\n"
                    "  → Add: ANTHROPIC_API_KEY=sk-ant-your-key-here"
                )
            self.llm_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            print(f"[Pipeline] LLM: {ANTHROPIC_MODEL} (Anthropic)")

        elif LLM_PROVIDER == "openai":
            if not OPENAI_API_KEY:
                raise EnvironmentError(
                    "[Pipeline] OPENAI_API_KEY is not set in your .env file.\n"
                    "  → Add: OPENAI_API_KEY=sk-your-key-here"
                )
            self.llm_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            print(f"[Pipeline] LLM: {OPENAI_MODEL} (OpenAI)")

        else:
            raise ValueError(
                f"[Pipeline] Unknown LLM_PROVIDER: '{LLM_PROVIDER}'\n"
                "  → Set LLM_PROVIDER to 'anthropic' or 'openai' in .env"
            )

    # ─────────────────────────────────────────────
    # INGESTION — Run once per document
    # ─────────────────────────────────────────────

    def ingest(self, force: bool = False) -> None:
        """
        Full ingestion pipeline: PDF → chunks → embeddings → ChromaDB.

        Steps:
          1. Check if already ingested (skip unless force=True)
          2. Run data preparation (load PDF, clean, chunk)
          3. Embed all chunks
          4. Store embeddings in ChromaDB

        Args:
          force : if True, clears the vector store and re-ingests from scratch.
                  Use this when you replace document.pdf with a new file.
        """
        print("\n" + "=" * 55)
        print("  RAG Pipeline — Ingestion")
        print("=" * 55)

        # Skip if already ingested (and not forcing)
        existing_count = self.vector_store.count()
        if existing_count > 0 and not force:
            print(f"[Pipeline] Vector store already has {existing_count} chunks.")
            print("  → Skipping ingestion. Use ingest(force=True) to re-ingest.")
            return

        if force and existing_count > 0:
            print("[Pipeline] Force re-ingestion requested — clearing vector store...")
            self.vector_store.clear()

        # Step 1: Data preparation (PDF → clean chunks)
        print("\n[Pipeline] Step 1/3 — Data preparation...")
        chunks = run_data_preparation()

        if not chunks:
            raise RuntimeError(
                "[Pipeline] No chunks produced. "
                "Check your PDF file at data/document.pdf"
            )

        # Step 2: Embed all chunks
        print(f"\n[Pipeline] Step 2/3 — Embedding {len(chunks)} chunks...")
        enriched_chunks = self.embedder.embed_chunks(chunks)

        # Step 3: Store in ChromaDB
        print("\n[Pipeline] Step 3/3 — Storing in vector database...")
        self.vector_store.add_chunks(enriched_chunks)

        print(f"\n✅ Ingestion complete — "
              f"{self.vector_store.count()} chunks ready for querying.")

    # ─────────────────────────────────────────────
    # QUERYING — Run for every user question
    # ─────────────────────────────────────────────

    def query(self, question: str) -> str:
        """
        Answer a question using the RAG pipeline.

        Steps:
          1. Retrieve top-3 relevant chunks from the vector store
          2. Format chunks into a context string
          3. Build the prompt (context + question → PROMPT_TEMPLATE)
          4. Send prompt to LLM
          5. Return the LLM's grounded answer

        Args:
          question : natural language question from the user

        Returns:
          answer string from the LLM, grounded in your document's content
        """
        print("\n" + "─" * 55)
        print(f"  Question: {question}")
        print("─" * 55)

        # Guard: must ingest before querying
        if self.vector_store.count() == 0:
            return (
                "⚠ The vector store is empty. "
                "Please run ingestion first:\n"
                "  pipeline.ingest()  or  python main.py --ingest"
            )

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(question)

        if not retrieved_chunks:
            return (
                "I could not find relevant information in the document "
                "to answer your question."
            )

        # Step 2: Format context
        context = self.retriever.format_context(retrieved_chunks)

        # Step 3: Build the prompt
        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        # Step 4: Call the LLM
        print(f"\n[Pipeline] Sending prompt to {LLM_PROVIDER.upper()} LLM...")
        answer = self._call_llm(prompt)

        # Step 5: Return answer
        print(f"\n[Pipeline] ✓ Answer received ({len(answer.split())} words)")
        return answer

    def _call_llm(self, prompt: str) -> str:
        """
        Send the prompt to the configured LLM and return the response text.
        Handles both Anthropic and OpenAI APIs transparently.
        """
        if LLM_PROVIDER == "anthropic":
            response = self.llm_client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=ANTHROPIC_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        elif LLM_PROVIDER == "openai":
            response = self.llm_client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=OPENAI_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def status(self) -> None:
        """Print a summary of the current pipeline state."""
        count = self.vector_store.count()
        print("\n── Pipeline Status ───────────────────────────────")
        print(f"  LLM Provider   : {LLM_PROVIDER.upper()}")
        print(f"  Embedding model: {self.embedder.model}")
        print(f"  Chunks in DB   : {count}")
        print(f"  Chunks file    : {CHUNKS_FILE} "
              f"({'exists' if CHUNKS_FILE.exists() else 'not found'})")
        print("──────────────────────────────────────────────────")


# ─────────────────────────────────────────────
# Quick smoke test — run this file directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.status()

    print("\n[Smoke Test] Checking ingestion status...")
    if pipeline.vector_store.count() == 0:
        print("  Vector store is empty.")
        print("  → Place your PDF at data/document.pdf then run:")
        print("    python main.py --ingest")
    else:
        print(f"  ✅ {pipeline.vector_store.count()} chunks ready.")
        print("  → Run python main.py to start querying!")