import json
import sys
from itertools import groupby, islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRACE = ROOT / "interference_system_prototype" / "input" / "bm25_13k_50.json"
total_turns = 0

sys.path.insert(0, str(ROOT / "config"))
from config import MODEL, MODEL_DIR


#? turns can be passed as list of int or an integer limit
def load(convs=None, turns=None, path=TRACE, tokenize=True):
    recs = json.loads(Path(path).read_text())
    out = []
    turn_counter = 0
    global total_turns
    
    if isinstance(turns, (list, tuple)):
        assert len(turns) == convs
    
    for conv_id, conv in islice(groupby(recs, key=lambda r: r.get("conv_id")), convs):
        limit = turns[conv_id] if isinstance(turns, (list, tuple)) else turns
        
        conv_turns = [
            {
                "id": f"{r['conv_id']}-{r.get('iter_id', 0)}",
                "conv": r["conv_id"],
                "prompt": r.get("prompt", ""),
                "max_tokens": r.get("out_token_size", 0),
                "in_token_size": r.get("in_token_size", 0),
                "out_token_size": r.get("out_token_size", 0),
            }
            for r in list(conv)[:limit]
        ]
        
        out.append(conv_turns)
        turn_counter += len(conv_turns)
    
    total_turns = turn_counter
    return _tokenize(out) if tokenize else out

def get_rec_turns():
    return total_turns

def _tokenize(recs):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=MODEL_DIR)
    bos = [tok.bos_token_id]
    out = []
    for conv in recs:
        turns = []
        for i, r in enumerate(conv):
            ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            turns.append({**r, "prompt": bos + ids, "ids": ids if i else bos + ids})
        out.append(turns)
    return out


if __name__ == "__main__":
    rec = load(convs=10, tokenize=False)
    for i in range(len(rec)):
        print(f"conv {i} : {len(rec[i])} turns")
