"""
config.py — Central configuration for the RAG pipeline.
All tunable parameters live here. Nothing is hardcoded elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load API keys from .env file

# ─────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
CHUNKS_FILE = DATA_DIR / "chunks.json"

# Place your PDF here before running the pipeline
PDF_PATH = DATA_DIR / "document.pdf"

# ─────────────────────────────────────────────
# LLM PROVIDER SELECTION
# Toggle between "anthropic" or "openai"
# ─────────────────────────────────────────────

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" | "openai"

# Anthropic settings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_MAX_TOKENS = 1024

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"
OPENAI_MAX_TOKENS = 1024

# ─────────────────────────────────────────────
# EMBEDDING MODEL
# Free, local, no API key needed
# ─────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~80MB, fast, high quality

# ─────────────────────────────────────────────
# TEXT CHUNKING (Sliding Window Strategy)
# ─────────────────────────────────────────────

CHUNK_SIZE = 150        # Max tokens/words per chunk
CHUNK_OVERLAP = 30     # Overlap between consecutive chunks (context continuity)
MIN_CHUNK_SIZE = 30     # Discard chunks shorter than this (noise removal)

# ─────────────────────────────────────────────
# VECTOR DATABASE (ChromaDB)
# ─────────────────────────────────────────────

CHROMA_COLLECTION_NAME = "enterprise_docs"
TOP_K_RESULTS = 3       # Number of chunks to retrieve per query

# ─────────────────────────────────────────────
# RAG PROMPT TEMPLATE
# {context} and {question} are filled at runtime
# ─────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions strictly based on the provided context.
If the answer is not found in the context, say "I don't have enough information to answer that."
Do not make up any information.

Context:
{context}

Question: {question}

Answer:"""
