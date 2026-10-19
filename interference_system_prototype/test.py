"""Replay the trace, pre-tokenized unless PRETOKENIZE is False, on the GPUs in ENGINES through the prototype alone or with the colocated and disaggregated baselines.

Run from this directory on a node with the GPUs in ENGINES and the conserve env active:
    python test.py
"""

import sys
from datetime import datetime
from pathlib import Path

for d in ("src", "output", "helpers"):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts" / d))

from load import load, get_rec_turns
from runs import OUTPUT_DIR, run_col, run_disagg, run_proto

ENGINES = {0: "kv_producer", 1: "kv_consumer"}
MAX_CONVS = 10
PRETOKENIZE = True


def test_all(recs, out):
    """Run the colocated and disaggregated conversation baselines and the prototype on recs' conversations into out."""
    print(f"{datetime.now():%H:%M:%S} [1/3] colocated: {get_rec_turns()} requests", flush=True)
    # run_col(recs, list(ENGINES), out / "colocated")
    for i, (name, run) in enumerate({"disagg": run_disagg, "prototype": run_proto}.items(), 2):
        print("-" * 60)
        print(f"{datetime.now():%H:%M:%S} [{i}/3] {name}: {get_rec_turns()} requests", flush=True)
        run(recs, ENGINES, out / name)

def test_col(recs, out):
    print(f"{datetime.now():%H:%M:%S} colocated: {get_rec_turns()} requests", flush=True)
    run_col(recs, ENGINES, out / "colocated")
    
def test_disagg(recs, out):
    print(f"{datetime.now():%H:%M:%S} disagg: {get_rec_turns()} requests", flush=True)
    run_disagg(recs, ENGINES, out / "disagg")

def test_proto(recs, out):
    print(f"{datetime.now():%H:%M:%S} prototype: {get_rec_turns()} requests", flush=True)
    run_proto(recs, ENGINES, out / "prototype")


if __name__ == "__main__":
    # test_col(load(convs=10, tokenize=PRETOKENIZE), OUTPUT_DIR / "test0")
    # test_disagg(load(convs=MAX_CONVS, tokenize=PRETOKENIZE), OUTPUT_DIR / "test_10")
    test_proto(load(convs=MAX_CONVS, tokenize=PRETOKENIZE), OUTPUT_DIR / "test0")
    # test_all(load(convs=MAX_CONVS, tokenize=PRETOKENIZE), OUTPUT_DIR / "test0")
