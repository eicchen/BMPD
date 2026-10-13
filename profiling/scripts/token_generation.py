"""Write a stream of random token IDs with a properties file, and cut decode and prefill prompts from it."""

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "config"))
from config import MODEL, MODEL_DIR

GENERATE_SEED = 0
LOAD_SEED = 0
ROLES = ("decode", "prefill")
DATA_FILE = ROOT / "profiling" / "input" / f"tokens_{MODEL.split('/')[-1]}.npy"


def generate(n_tokens):
    """Write n_tokens random token IDs plus a properties file; return the data file path."""
    vocab_size = AutoTokenizer.from_pretrained(MODEL, cache_dir=str(MODEL_DIR)).vocab_size
    tokens = np.random.default_rng(GENERATE_SEED).integers(vocab_size, size=n_tokens, dtype=np.int32)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(DATA_FILE, tokens)
    props = {
        "generate_seed": GENERATE_SEED,
        "tokenizer": MODEL,
        "vocab_size": vocab_size,
        "numpy_version": np.__version__,
        "n_tokens": n_tokens,
        "data_file": DATA_FILE.name,
        "sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    DATA_FILE.with_suffix(".properties").write_text("".join(f"{k}={v}\n" for k, v in props.items()))
    return DATA_FILE


def load(decode, prefill, load_seed=LOAD_SEED):
    """decode/prefill: {length: count}. Cut consecutive prompts from the stream, starting at a seeded offset."""
    stream = np.load(DATA_FILE, mmap_mode="r")
    segs = [
        (role, L, i)
        for role, pool in zip(ROLES, (decode, prefill))
        for L, n in pool.items()
        for i in range(n)
    ]
    total = sum(s[1] for s in segs)
    if total > len(stream):
        raise ValueError(f"need {total} tokens, stream has {len(stream)}")
    o = np.random.default_rng(load_seed).integers(len(stream) - total + 1)
    recs = []
    for role, L, i in segs:
        rec = {
            "id": f"{role}-{L}-{i:04d}",
            "role": role,
            "length": L,
            "token_ids": stream[o:o + L].tolist(),
        }
        recs.append(rec)
        o += L
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, required=True)
    args = ap.parse_args()
    generate(args.n_tokens)


if __name__ == "__main__":
    main()
