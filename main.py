"""
main.py — Command-line entry point for the RAG pipeline.

Location: rag_enterprise/main.py

This is the ONLY file you need to run directly.
It provides a clean CLI interface with three modes:

  MODE 1 — Ingest your document (run once):
    python main.py --ingest

  MODE 2 — Ask a single question:
    python main.py --query "What is the leave policy?"

  MODE 3 — Interactive chat loop (ask multiple questions):
    python main.py

  BONUS — Re-ingest after replacing document.pdf:
    python main.py --ingest --force
"""

import argparse
import sys

from rag_pipeline import RAGPipeline


# ─────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────

BANNER = """
╔═══════════════════════════════════════════════════════╗
║       RAG System for Enterprise Data                  ║
║       Retrieval-Augmented Generation Pipeline         ║
╚═══════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────
# INTERACTIVE CHAT LOOP
# ─────────────────────────────────────────────

def run_interactive_loop(pipeline: RAGPipeline) -> None:
    """
    Start an interactive Q&A session in the terminal.

    The user can ask as many questions as they want.
    Type 'exit', 'quit', or press Ctrl+C to stop.
    Type 'status' to see pipeline stats.
    Type 'help' to see available commands.
    """
    print("\n💬 Interactive RAG Chat")
    print("   Ask questions about your document.")
    print("   Commands: 'exit' to quit | 'status' for stats | 'help' for commands")
    print("─" * 55)

    while True:
        try:
            # Prompt the user for input
            user_input = input("\n🧑 You: ").strip()

            # Skip empty input
            if not user_input:
                continue

            # ── Built-in commands ──────────────────────────
            if user_input.lower() in ("exit", "quit", "q"):
                print("\n👋 Goodbye! RAG session ended.")
                break

            elif user_input.lower() == "status":
                pipeline.status()
                continue

            elif user_input.lower() == "help":
                print("\n  Available commands:")
                print("  exit / quit / q  — end the session")
                print("  status           — show pipeline stats")
                print("  help             — show this message")
                print("  (anything else)  — ask a question about your document")
                continue

            # ── Answer the question ────────────────────────
            print("\n🤖 Assistant: Thinking...\n")
            answer = pipeline.query(user_input)

            print(f"\n🤖 Assistant:\n")
            print(f"   {answer}\n")
            print("─" * 55)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\n👋 Session interrupted. Goodbye!")
            break

        except Exception as e:
            print(f"\n⚠ Error during query: {e}")
            print("  Please try again or type 'exit' to quit.")


# ─────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="RAG System for Enterprise Data — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                     # Interactive chat
  python main.py --ingest                            # Ingest document
  python main.py --ingest --force                    # Re-ingest (clears DB)
  python main.py --query "What is the HR policy?"   # Single question
  python main.py --status                            # Show pipeline status
        """,
    )

    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest the PDF document into the vector database (run once)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if the vector store already has data",
    )
    parser.add_argument(
        "--query",
        type=str,
        metavar="QUESTION",
        help="Ask a single question and print the answer",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show the current pipeline status and exit",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    """Entry point — parse args and run the appropriate pipeline mode."""
    print(BANNER)
    args = parse_args()

    # Initialise the pipeline (loads models, connects to ChromaDB)
    try:
        pipeline = RAGPipeline()
    except EnvironmentError as e:
        print(f"\n❌ Setup Error:\n  {e}")
        print("\n  → Create a .env file in rag_enterprise/ with your API key.")
        print("  → See .env.example for the template.")
        sys.exit(1)

    # ── Mode 1: Status check ──────────────────────────
    if args.status:
        pipeline.status()
        sys.exit(0)

    # ── Mode 2: Ingest ────────────────────────────────
    if args.ingest:
        try:
            pipeline.ingest(force=args.force)
        except FileNotFoundError as e:
            print(f"\n❌ File Error: {e}")
            print("  → Place your PDF at rag_enterprise/data/document.pdf")
            sys.exit(1)
        except RuntimeError as e:
            print(f"\n❌ Ingestion Error: {e}")
            sys.exit(1)
        print("\n✅ Ingestion complete. You can now run:")
        print("   python main.py                          # Interactive chat")
        print("   python main.py --query 'Your question'  # Single question")
        sys.exit(0)

    # ── Mode 3: Single query ──────────────────────────
    if args.query:
        answer = pipeline.query(args.query)
        print(f"\n🤖 Answer:\n\n   {answer}\n")
        sys.exit(0)

    # ── Mode 4: Interactive loop (default) ────────────
    # Check vector store has data before starting the loop
    if pipeline.vector_store.count() == 0:
        print("⚠  The vector store is empty — no document has been ingested yet.")
        print("\n   Step 1: Place your PDF at:  rag_enterprise/data/document.pdf")
        print("   Step 2: Run:                 python main.py --ingest")
        print("   Step 3: Then run:            python main.py\n")
        sys.exit(0)

    run_interactive_loop(pipeline)


if __name__ == "__main__":
    main()