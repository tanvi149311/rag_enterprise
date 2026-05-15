# RAG System for Enterprise Data

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline built in Python that answers questions about your private documents — accurately, without hallucinations.

---

## What is RAG?

Generic AI models like ChatGPT don't know your private company data, and they frequently "hallucinate" incorrect answers. RAG solves both problems:

1. **Retrieval** — Instantly searches your private document for the most relevant paragraphs
2. **Augmented Generation** — Feeds only those paragraphs to the LLM as context
3. **Result** — Accurate, grounded answers based exclusively on your data

```
Your PDF  →  Clean  →  Chunk  →  Embed  →  ChromaDB
                                               ↑
User Question  →  Embed  →  Search  →  Top-3 Chunks  →  LLM  →  Answer ✅
```

---

## Project Structure

```
rag_enterprise/
├── config.py              # All settings (models, paths, chunk params)
├── data_preparation.py    # PDF loader + NLP cleaner + sliding window chunker
├── embedder.py            # Converts text → vector embeddings (local, free)
├── vector_store.py        # ChromaDB wrapper for storing & searching embeddings
├── retriever.py           # Finds top-3 relevant chunks for a query
├── rag_pipeline.py        # Orchestrates full ingestion + query pipeline
├── main.py                # CLI entry point — run this file
├── test_chunker.py        # Unit tests for the chunker and cleaner
├── requirements.txt       # All Python dependencies
├── .env.example           # API key template (copy → .env, add real keys)
├── data/
│   ├── document.pdf       # ← Place YOUR PDF here
│   └── chunks.json        # Auto-generated after ingestion
└── chroma_db/             # Auto-generated vector database
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| PDF Extraction | PyMuPDF (fitz) |
| Text Cleaning | NLTK + regex |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB (local, persistent) |
| LLM — Option A | Anthropic Claude (`claude-sonnet-4-20250514`) |
| LLM — Option B | OpenAI GPT-4o |
| Testing | pytest |

---

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.9+ | `python3 --version` |
| pip | latest | `pip --version` |
| Git | any | `git --version` |

---

## Setup Guide

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag_enterprise.git
cd rag_enterprise
```

### Step 2 — Create and activate a virtual environment

```bash
python3 -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> First run takes 2–3 minutes — downloads PyTorch and the embedding model (~80MB).

### Step 4 — Configure your API key

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```env
# To use Anthropic Claude (recommended — free credits on signup):
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here

# To use OpenAI GPT-4o (requires billing):
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-your-actual-key-here
```

**Get your API key:**
- Anthropic → [console.anthropic.com](https://console.anthropic.com) → API Keys
- OpenAI → [platform.openai.com](https://platform.openai.com) → API Keys

### Step 5 — Add your PDF document

```bash
# Copy your PDF into the data folder and rename it
cp /path/to/your/document.pdf data/document.pdf
```

---

## Running the Pipeline

### Step 1 — Ingest your document (run once)

```bash
python main.py --ingest
```

Expected output:
```
[1/3] Loading PDF: document.pdf
      ✓ Extracted text from 42 pages (98,432 characters)
[2/3] Cleaning text...
      ✓ Cleaned text: 87,210 characters remaining
[3/3] Chunking text with sliding window strategy...
      ✓ Created 186 chunks (avg 487 words/chunk)
✅ Ingestion complete — 186 chunks ready for querying.
```

### Step 2 — Ask questions (interactive mode)

```bash
python main.py
```

```
💬 Interactive RAG Chat
   Ask questions about your document.
──────────────────────────────────────────────────

🧑 You: What is the company's annual leave policy?

🤖 Assistant:
   Based on the document, employees are entitled to 20 days
   of paid annual leave per year, which accrues monthly...
```

### Other run modes

```bash
# Single question (non-interactive)
python main.py --query "What are the compliance requirements?"

# Check pipeline status
python main.py --status

# Re-ingest after replacing document.pdf
python main.py --ingest --force

# Show all available options
python main.py --help
```

---

## Running the Tests

```bash
pytest test_chunker.py -v
```

Expected output:
```
test_chunker.py::TestCleanText::test_removes_page_numbers    PASSED
test_chunker.py::TestCleanText::test_removes_urls            PASSED
...
23 passed in 1.2s
```

---

## Switching LLM Providers

No code changes needed — just edit `.env`:

```env
# Switch to OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

Then run your query as normal. The pipeline adapts automatically.

---

## How It Works — Deep Dive

### Part 1: Data Preparation

```
PDF
 └─ load_pdf()         Extract text page by page (PyMuPDF)
     └─ clean_text()   Remove noise: page numbers, URLs, unicode artifacts
         └─ chunk_text() Split into 500-word overlapping chunks (NLTK sentences)
             └─ chunks.json saved to disk
```

**Sliding Window Strategy:**
- Each chunk is up to 500 words
- Consecutive chunks share 100 words of overlap
- Sentences are never split mid-way (NLTK sentence tokenizer)
- Chunks under 50 words are discarded as noise

### Part 2: Ingestion & Querying

```
INGESTION (once per document):
chunks.json → Embedder → 384-dim vectors → ChromaDB

QUERYING (every question):
Question → embed → cosine search → top-3 chunks
        → PROMPT_TEMPLATE → LLM → grounded answer
```

**Anti-hallucination mechanism:**
The LLM prompt explicitly instructs the model:
> "Answer only from the provided context. If the answer is not found, say so."

---

## Pushing to GitHub

### First time setup

```bash
# 1. Create a new repo on github.com (do NOT initialise with README)

# 2. Inside your project folder:
git init
git add .
git commit -m "Initial commit: RAG pipeline for enterprise data"

# 3. Connect to GitHub and push
git remote add origin https://github.com/YOUR_USERNAME/rag_enterprise.git
git branch -M main
git push -u origin main
```

### Subsequent pushes

```bash
git add .
git commit -m "Your descriptive commit message"
git push
```

### What gets pushed vs ignored

The `.gitignore` ensures sensitive and generated files are never committed:

| Ignored (stays local) | Pushed to GitHub |
|-----------------------|-----------------|
| `.env` (your API keys) | `.env.example` (safe template) |
| `chroma_db/` (vector DB) | All `.py` source files |
| `data/document.pdf` | `requirements.txt` |
| `venv/` | `README.md` |
| `__pycache__/` | `test_chunker.py` |

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `PDF not found` | Place your PDF at `data/document.pdf` |
| `API key not set` | Check your `.env` file has the correct key |
| `No results returned` | Run `python main.py --ingest` first |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with venv active |
| `Distance threshold exceeded` | Your question may be out of scope of the document |
| Want to use a new PDF | Run `python main.py --ingest --force` |

---

## License

MIT License — free to use, modify, and distribute.