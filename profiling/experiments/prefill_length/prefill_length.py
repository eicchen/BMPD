"""Experiment 1: sweep prefill length at a fixed arrival rate over a decode background.

Run from this directory on a GPU node with the conserve env active:
    python prefill_length.py
    python -c "from prefill_length import sweep; sweep(False)"
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MAX_MODEL_LEN, OUTPUT_DIR, launch_vllm, run_decode_only, run_point, run_prefill_only

EXPERIMENT = Path(__file__).stem
PREFILL_LENGTHS = [256, 512, 1024, 2048, 4096, 8192, 16384]
ARRIVAL_RATE = 1.0
PREFILL_COUNT = 32
DECODE_COUNT = 32
DECODE_LENGTH = 1024
DECODE_OUTPUT = 2048
CHUNKED_BATCH_TOKENS = 2048


def sweep(chunked):
    """Run the decodes alone, then each prefill length alone and over the decodes, on one server."""
    log_dir = OUTPUT_DIR / EXPERIMENT / "data" / ("chunked" if chunked else "no_chunked")
    flags = [
        "--enable-chunked-prefill" if chunked else "--no-enable-chunked-prefill",
        "--max-num-batched-tokens",
        str(CHUNKED_BATCH_TOKENS if chunked else MAX_MODEL_LEN),
        "--no-enable-prefix-caching",
    ]
    print(f"{datetime.now():%H:%M:%S} Launching VLLM...")
    proc = launch_vllm(log_dir, flags)
    try:
        print(f"{datetime.now():%H:%M:%S} chunked={chunked} decode_only", flush=True)
        run_decode_only(log_dir, DECODE_COUNT, DECODE_LENGTH, DECODE_OUTPUT)
        for i, length in enumerate(PREFILL_LENGTHS, 1):
            print(f"{datetime.now():%H:%M:%S} [{i}/{len(PREFILL_LENGTHS)}] chunked={chunked} prefill_length={length} arrival_rate={ARRIVAL_RATE:g}", flush=True)
            run_prefill_only(log_dir, length, ARRIVAL_RATE, PREFILL_COUNT)
            run_point(log_dir, length, ARRIVAL_RATE, PREFILL_COUNT, DECODE_COUNT, DECODE_LENGTH, DECODE_OUTPUT)
    finally:
        print(f"{datetime.now():%H:%M:%S} Terminating VLLM...")
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    sweep(True)
    sweep(False)
