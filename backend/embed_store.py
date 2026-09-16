"""
Step 3: Embed all chunks and store in FAISS.

Supports two backends — pick one:
  (A) OpenAI text-embedding-3-small  [requires OPENAI_API_KEY, ~$0.004]
  (B) BAAI/bge-small-en-v1.5         [free, fully local, no API key needed]

Usage:
    # Free local (default):
    python3 embed_store.py

    # OpenAI:
    OPENAI_API_KEY="sk-..." python3 embed_store.py --backend openai

Input:  chunks/all_chunks.jsonl
Output: faiss_index/index.faiss
        faiss_index/metadata.jsonl
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import faiss

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks", "all_chunks.jsonl")
INDEX_DIR   = os.path.join(BASE_DIR, "faiss_index")
INDEX_FILE  = os.path.join(INDEX_DIR, "index.faiss")
META_FILE   = os.path.join(INDEX_DIR, "metadata.jsonl")


# ── Backend: local BGE ─────────────────────────────────────────────────────

BGE_MODEL_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bge_model")
BGE_DIM        = 384
BGE_BATCH_SIZE = 64     # sentence-transformers handles batching internally

def embed_bge(texts, model):
    """Embed a list of texts using BGE. Returns numpy (n, 384) float32."""
    # BGE requires a query prefix for retrieval tasks
    # For document chunks (passage side), no prefix needed
    vecs = model.encode(texts, batch_size=BGE_BATCH_SIZE,
                        show_progress_bar=False,
                        normalize_embeddings=True)   # L2-normalize in place
    return vecs.astype(np.float32)


# ── Backend: OpenAI ────────────────────────────────────────────────────────

OPENAI_MODEL     = "text-embedding-3-small"
OPENAI_DIM       = 1536
OPENAI_BATCH     = 100
OPENAI_RETRY_SEC = 5

def embed_openai(texts, client):
    """Embed a batch of texts via OpenAI API. Returns list of vectors."""
    for attempt in range(4):
        try:
            resp = client.embeddings.create(model=OPENAI_MODEL, input=texts)
            return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        except Exception as e:
            if attempt < 3:
                wait = OPENAI_RETRY_SEC * (attempt + 1)
                print(f"  [retry {attempt+1}] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                raise


# ── Core ───────────────────────────────────────────────────────────────────

def load_chunks():
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_and_save(chunks, embed_fn, dim):
    n = len(chunks)
    texts = [c["text"] for c in chunks]

    print(f"Embedding {n} chunks (dim={dim})...")
    all_vecs = embed_fn(texts)

    # Guarantee shape and dtype
    matrix = np.array(all_vecs, dtype=np.float32).reshape(n, dim)

    # If not already normalized (OpenAI path), normalize now
    faiss.normalize_L2(matrix)

    # IndexFlatIP = exact cosine similarity search (after L2 normalization)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    print(f"FAISS index: {index.ntotal} vectors")

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    print(f"Saved → {INDEX_FILE}")

    with open(META_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved → {META_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["bge", "openai"], default="bge",
                        help="Embedding backend (default: bge)")
    args = parser.parse_args()

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks\n")

    if args.backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: set OPENAI_API_KEY first.")
            sys.exit(1)
        from openai import OpenAI
        client  = OpenAI(api_key=api_key)
        batches = [chunks[i:i+OPENAI_BATCH] for i in range(0, len(chunks), OPENAI_BATCH)]
        all_vecs = []
        for i, batch in enumerate(batches):
            vecs = embed_openai([c["text"] for c in batch], client)
            all_vecs.extend(vecs)
            print(f"  {min((i+1)*OPENAI_BATCH, len(chunks))}/{len(chunks)}", end="\r")
        embed_fn = lambda _: np.array(all_vecs, dtype=np.float32)
        dim = OPENAI_DIM

    else:  # bge (default)
        from sentence_transformers import SentenceTransformer
        print(f"Loading model: {BGE_MODEL_NAME}  (first run downloads ~130 MB)")
        model = SentenceTransformer(BGE_MODEL_NAME)
        embed_fn = lambda texts: embed_bge(texts, model)
        dim = BGE_DIM

    build_and_save(chunks, embed_fn, dim)
    print("\nDone. Run:  python3 query.py  to test retrieval.")


if __name__ == "__main__":
    main()
