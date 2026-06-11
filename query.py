"""
Step 4: RAG query interface — FAISS retrieval + Claude API answer.

Usage:
    # Interactive CLI:
    python3 query.py

    # Single question:
    python3 query.py --question "What are the side effects of metformin?"

    # Filter by drug:
    python3 query.py --drug metformin --question "Can I take it with alcohol?"

    # Adjust how many chunks to retrieve (default: 5):
    python3 query.py --top-k 8

Requirements:
    pip install anthropic sentence-transformers faiss-cpu numpy
    ANTHROPIC_API_KEY must be set in the environment.
"""

import os
import sys
import json
import argparse
import numpy as np
import faiss

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE  = os.path.join(BASE_DIR, "faiss_index", "index.faiss")
META_FILE   = os.path.join(BASE_DIR, "faiss_index", "metadata.jsonl")
BGE_DIR     = os.path.join(BASE_DIR, "bge_model")

DEFAULT_TOP_K   = 5
CLAUDE_MODEL    = "claude-opus-4-8"
MAX_CONTEXT_CHARS = 12_000   # ~3000 tokens of context; well within Claude's window

SYSTEM_PROMPT = """\
You are a clinical pharmacist assistant helping patients understand their prescription medications.
Your answers must be based ONLY on the FDA drug label excerpts provided below.
Rules:
1. If the answer is in the excerpts, answer clearly and cite the source section.
2. If the excerpts do not contain enough information to answer, say so explicitly — do NOT invent or extrapolate.
3. Always include a safety reminder at the end: patients should consult their prescriber or pharmacist before making any changes.
4. Keep the answer concise and patient-friendly (avoid excessive jargon).
"""


# ── Loading ─────────────────────────────────────────────────────────────────

def load_index_and_meta():
    if not os.path.exists(INDEX_FILE):
        sys.exit(f"ERROR: FAISS index not found at {INDEX_FILE}\n"
                 "Run  python3 embed_store.py  first.")
    index = faiss.read_index(INDEX_FILE)

    with open(META_FILE, encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f if line.strip()]

    if index.ntotal != len(metadata):
        sys.exit(f"ERROR: index has {index.ntotal} vectors but metadata has {len(metadata)} rows.")

    return index, metadata


def load_bge_model():
    if not os.path.isdir(BGE_DIR):
        sys.exit(f"ERROR: BGE model not found at {BGE_DIR}\n"
                 "Run  python3 download_model.py  first.")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_DIR)
    return model


# ── Retrieval ────────────────────────────────────────────────────────────────

def embed_query(question: str, model) -> np.ndarray:
    # BGE uses a query prefix for retrieval (passage side has no prefix)
    prefixed = f"Represent this sentence: {question}"
    vec = model.encode([prefixed], normalize_embeddings=True)
    return vec.astype(np.float32)


def retrieve(question: str, model, index, metadata,
             top_k: int = DEFAULT_TOP_K,
             drug_filter: str | None = None) -> list[dict]:
    q_vec = embed_query(question, model)

    # Search more candidates when drug_filter is active, then post-filter
    search_k = top_k * 6 if drug_filter else top_k
    scores, indices = index.search(q_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx]
        if drug_filter and chunk.get("drug", "").lower() != drug_filter.lower():
            continue
        results.append({**chunk, "_score": float(score)})
        if len(results) >= top_k:
            break

    return results


# ── Prompt assembly ───────────────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    parts = []
    total = 0
    for c in chunks:
        header = f"[{c['drug'].upper()} — {c['section']}]"
        body   = c["text"].strip()
        block  = f"{header}\n{body}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)


def build_messages(question: str, context: str) -> list[dict]:
    user_content = (
        "Drug Label Excerpts:\n"
        "──────────────────────────────────────────────\n"
        f"{context}\n"
        "──────────────────────────────────────────────\n\n"
        f"Patient question: {question}"
    )
    return [{"role": "user", "content": user_content}]


# ── Claude call ───────────────────────────────────────────────────────────────

def ask_claude(question: str, context: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=build_messages(question, context),
    )
    return response.content[0].text


# ── Main answer function ──────────────────────────────────────────────────────

def answer(question: str,
           model,
           index,
           metadata,
           api_key: str,
           top_k: int = DEFAULT_TOP_K,
           drug_filter: str | None = None,
           verbose: bool = False) -> str:

    chunks = retrieve(question, model, index, metadata, top_k, drug_filter)
    if not chunks:
        return ("No relevant drug label excerpts found"
                + (f" for drug '{drug_filter}'" if drug_filter else "")
                + ". Please rephrase your question or remove the drug filter.")

    if verbose:
        print(f"\n[Retrieved {len(chunks)} chunks]")
        for i, c in enumerate(chunks, 1):
            print(f"  {i}. [{c['drug']}] {c['section']}  (score={c['_score']:.3f})")

    context = build_context(chunks)
    return ask_claude(question, context, api_key)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedBuddy RAG — ask questions about FDA drug labels")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Question to ask (if omitted, enters interactive mode)")
    parser.add_argument("--drug", "-d", type=str, default=None,
                        help="Restrict search to a specific drug (e.g. metformin)")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print retrieved chunk headers before the answer")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY environment variable is not set.")

    print("Loading FAISS index...", end=" ", flush=True)
    index, metadata = load_index_and_meta()
    print(f"{index.ntotal} vectors")

    print("Loading BGE model...", end=" ", flush=True)
    model = load_bge_model()
    print("ready\n")

    def run_query(question):
        print("Searching...", end=" ", flush=True)
        resp = answer(question, model, index, metadata, api_key,
                      top_k=args.top_k, drug_filter=args.drug, verbose=args.verbose)
        print("\n" + "═" * 60)
        print(resp)
        print("═" * 60)

    if args.question:
        run_query(args.question)
    else:
        # Interactive mode
        drug_hint = f" (filtered to: {args.drug})" if args.drug else ""
        print(f"MedBuddy Drug Label Q&A{drug_hint}")
        print("Type your question and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                q = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not q:
                continue
            if q.lower() in {"quit", "exit", "q"}:
                print("Exiting.")
                break
            run_query(q)
            print()


if __name__ == "__main__":
    main()
