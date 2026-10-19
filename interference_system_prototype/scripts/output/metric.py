"""Parse a run's engine logs into per-GPU-request fields, per-step rows and a run summary, and write them into the run's directory."""

import json
import re
from bisect import bisect_right
from datetime import datetime
from itertools import accumulate

SUFFIX = re.compile(r"-[0-9a-f]{8}$")
HOLD = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\].*\[conserve_pd\] hold (?:released|timeout) req=(\S+) .*?waited=([\d.]+)s")
ITERATION = re.compile(r"Iteration\(\d+\): (\d+) context requests, (\d+) context tokens, (\d+) generation requests, (\d+) generation tokens")
COUNTS = ("ctx_reqs", "ctx_tokens", "gen_reqs", "gen_tokens")
GPU_REQUEST_FIELDS = ("arrived", "first_step", "hold_start", "hold_end", "mixed_wait")


def _jsonl(path):
    """Records of a JSONL file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ts(iso):
    """Epoch seconds of an ISO timestamp."""
    return datetime.fromisoformat(iso).timestamp()


def _steps(log_dir, log):
    """Engine steps from log_dir's core.jsonl, with the context/generation counts of each running step from its server log."""
    steps, start = [], None
    for rec in _jsonl(log_dir / "core.jsonl"):
        if rec["event"] == "step_start":
            start = _ts(rec["timestamp"])
        else:
            steps.append(
                {
                    "start": start,
                    "end": _ts(rec["timestamp"]),
                    "ids": [SUFFIX.sub("", rid) for rid in rec["executed_request_ids"]],
                    **dict.fromkeys(COUNTS, 0),
                    "kv_usage": rec["kv_cache_usage"],
                },
            )
    for step, counts in zip([s for s in steps if s["ids"]], ITERATION.findall(log), strict=True):
        step.update(zip(COUNTS, map(int, counts)))
    return steps


def _gpu_logs(log_dir):
    """Write log_dir/steps.jsonl; return {request id: GPU request fields} and the steps that ran both context and generation work."""
    log = (log_dir / "server.log").read_text()
    steps = _steps(log_dir, log)
    arrived = {rec["request_id"]: _ts(rec["timestamp"]) for rec in _jsonl(log_dir / "engine.jsonl") if rec["event"] == "request_start"}
    first = {}
    for s in steps:
        for rid in s.pop("ids"):
            first.setdefault(rid, s["start"])
    arrivals, firsts = sorted(arrived.values()), sorted(first.values())
    for s in steps:
        s["waiting"] = bisect_right(arrivals, s["start"]) - bisect_right(firsts, s["start"])
    (log_dir / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    mixed = [s for s in steps if s["ctx_reqs"] and s["gen_reqs"]]
    starts = [s["start"] for s in mixed]
    total = list(accumulate((s["end"] - s["start"] for s in mixed), initial=0.0))

    def mixed_until(t):
        i = bisect_right(starts, t)
        return total[i] - (max(0, mixed[i - 1]["end"] - t) if i else 0)

    holds = {}
    for ts, rid, waited in HOLD.findall(log):
        end = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f").timestamp()
        holds[SUFFIX.sub("", rid)] = {"hold_start": end - float(waited), "hold_end": end}
    gpu_requests = {}
    for rid, a in arrived.items():
        f = first.get(rid)
        gpu_requests[rid] = {
            "arrived": a,
            "first_step": f,
            **holds.get(rid, {"hold_start": None, "hold_end": None}),
            "mixed_wait": mixed_until(f) - mixed_until(a) if f else None,
        }
    return gpu_requests, mixed


def annotate(rows, engines, out):
    """Add each GPU request's log fields to rows from the engine logs in out, and write requests.jsonl, steps.jsonl and summary.properties."""
    summary = {"kv_producer_colocated": sum(len(r["gpu_requests"]) == 1 and engines[r["gpu_requests"][0]["gpu"]] == "kv_producer" for r in rows)}
    ends = {}
    for gpu in engines:
        gpu_requests, mixed = _gpu_logs(out / f"gpu{gpu}")
        ends |= {(gpu, rid): gpu_request for rid, gpu_request in gpu_requests.items()}
        summary[f"gpu{gpu}.mixed_steps"] = len(mixed)
        summary[f"gpu{gpu}.mixed_s"] = round(sum(s["end"] - s["start"] for s in mixed), 3)
    for r in rows:
        for gpu_request in r["gpu_requests"]:
            gpu_request |= ends.get((gpu_request["gpu"], f"cmpl-{r['id']}-0"), dict.fromkeys(GPU_REQUEST_FIELDS))
    (out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.properties").write_text("".join(f"{k}={v}\n" for k, v in summary.items()))
