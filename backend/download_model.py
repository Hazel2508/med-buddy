"""Download BGE's ONNX model and tokenizer for the public API."""
import os
import requests

MIRROR   = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
REPO     = "BAAI/bge-small-en-v1.5"
OUT_DIR  = os.environ.get(
    "MEDBUDDY_BGE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "bge_model"),
)

# Files needed for ONNX inference (skip PyTorch weights)
NEEDED = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json",
    "onnx/model.onnx",          # CPU inference without PyTorch on Render Free
]

session = requests.Session()
session.headers["User-Agent"] = "python-requests"


def download_file(rel_path):
    dest = os.path.join(OUT_DIR, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        print(f"  skip {rel_path} ({size//1024} KB already)")
        return

    url = f"{MIRROR}/{REPO}/resolve/main/{rel_path}"
    print(f"  downloading {rel_path} ...", end=" ", flush=True)
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  downloading {rel_path} ... {done*100//total}%", end="", flush=True)
    print(f"\r  {rel_path} done ({done//1024} KB)          ")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading {REPO} → {OUT_DIR}\n")
    for f in NEEDED:
        download_file(f)
    print(f"\nAll ONNX model files ready in {OUT_DIR}")


if __name__ == "__main__":
    main()
