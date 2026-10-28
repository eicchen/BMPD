"""Mixes prefill and decode requests served from stored KV and prefill requests on one colocated engine, swept over the decode ratio.

Run from this directory on a GPU node with the conserve env active. run and sweep launch their own servers; with launch_engine=False, as in python pd_mix.py, they use those of python launch_interference.py running in another terminal on the same node:
    python pd_mix.py
    python -c "import pd_mix; from launch_interference import ping; ping()"
    python -c "from pd_mix import *; run('test', DECODE_LENGTHS, PREFILL_LENGTHS, 16, ARRIVAL_RATE, 0.5)"
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import GPU, KV_DIR, OUTPUT_ROOT, _log, base_properties, check, engine_logs, load_decode_file, log_indent, send, send_times, servers, wipe_engine, write_decode_file, write_properties
from metric import parse_engine_logs
from token_generation import load_standard

DECODE_LENGTHS = (2048, 512)
PREFILL_LENGTHS = (2048, 1)
REQUEST_COUNT = 128
WARMUP_REQUEST_COUNT = 16
ARRIVAL_RATE = 2.0
DECODE_RATIOS = (0, 0.25, 0.5, 0.75, 1)
MIX_SEED = 0
PROMPT_SEED = {"decode": 0, "prefill": 1}
OUTPUT_DIR = OUTPUT_ROOT / Path(__file__).stem


# Requests (decode and prefill requests, and the mix)


def build_decode_requests(prompt_length, output_length, request_count):
    """Decode requests: prompts whose KV is in the decode files, each generating output_length tokens."""
    return [
        {"id": f"decode-{index:04d}", "request_kind": "decode", "prompt": prompt_record["token_ids"], "max_tokens": output_length}
        for index, prompt_record in enumerate(load_standard({prompt_length: request_count}, PROMPT_SEED["decode"]))
    ]


def build_prefill_requests(prompt_length, output_length, request_count):
    """Prefill requests: prompts with no stored KV, each generating output_length tokens."""
    return [
        {"id": f"prefill-{index:04d}", "request_kind": "prefill", "prompt": prompt_record["token_ids"], "max_tokens": output_length}
        for index, prompt_record in enumerate(load_standard({prompt_length: request_count}, PROMPT_SEED["prefill"]))
    ]


def mix(decode_requests, prefill_requests, decode_ratio, arrival_rate, served_kinds=("decode", "prefill")):
    """One run's request list: each slot is a decode or prefill request picked at decode_ratio, with a Poisson send time at arrival_rate (all at time 0 if arrival_rate is None); solo runs keep only served_kinds."""
    request_count = len(decode_requests)
    decode_slots = set(np.random.default_rng(MIX_SEED).permutation(request_count)[: round(decode_ratio * request_count)].tolist())
    slot_requests = (decode_requests[slot] if slot in decode_slots else prefill_requests[slot] for slot in range(request_count))
    return [
        {**request, "send_time": float(send_time)}
        for request, send_time in zip(slot_requests, send_times(arrival_rate, request_count))
        if request["request_kind"] in served_kinds
    ]


# Run (single runs and sweeps)


def _serve_run(
    output_dir,
    server_dir,
    decode_lengths,
    prefill_lengths,
    request_count,
    arrival_rate,
    decode_ratio,
    served_kinds,
    gpu,
):
    """One run on servers already running with logs in server_dir: wipe them, load the decode KV, send the requests, then save this run's slice of the engine logs, the per-request timings and run.properties to output_dir."""
    mixed_requests = mix(
        build_decode_requests(*decode_lengths, request_count),
        build_prefill_requests(*prefill_lengths, request_count),
        decode_ratio,
        arrival_rate,
        served_kinds,
    )
    decode_requests = [request for request in mixed_requests if request["request_kind"] == "decode"]
    wipe_engine(gpu)
    with engine_logs(server_dir, output_dir, gpu):
        _log(f"Loading KV of {len(decode_requests)} decode prompts into L1...")
        found_chunks, total_chunks = load_decode_file(decode_requests)
        _log(f"Loaded {found_chunks}/{total_chunks} chunks")
        if found_chunks < total_chunks:
            raise RuntimeError(f"Only {found_chunks}/{total_chunks} decode chunks found in {KV_DIR}")
        request_results = asyncio.run(send(mixed_requests, gpu))
    parse_engine_logs(request_results, {gpu: "kv_both"}, output_dir)
    write_properties(
        output_dir / "run.properties",
        {
            "decode_prompt_length": decode_lengths[0],
            "decode_output_length": decode_lengths[1],
            "prefill_prompt_length": prefill_lengths[0],
            "prefill_output_length": prefill_lengths[1],
            "request_count": request_count,
            "arrival_rate": arrival_rate,
            "decode_ratio": decode_ratio,
            "served_kinds": ",".join(served_kinds),
            "mix_seed": MIX_SEED,
            "decode_prompt_seed": PROMPT_SEED["decode"],
            "decode_kv_dir": KV_DIR,
            "found_chunks": found_chunks,
            "total_chunks": total_chunks,
            **check(request_results, ("decode", "prefill")),
            **base_properties(gpu, server_dir),
        },
    )
    return request_results


def run(
    experiment_name,
    decode_lengths,
    prefill_lengths,
    request_count,
    arrival_rate,
    decode_ratio,
    served_kinds=("decode", "prefill"),
    gpu=GPU,
    launch_engine=True,
):
    """One run on fresh servers, or those of launch_interference.py without launch_engine: decode requests (KV loaded from the files) and prefill requests (full prefill) share one engine at decode_ratio and arrival_rate; logs, per-request timings and run.properties go to OUTPUT_DIR/experiment_name/data."""
    output_dir = OUTPUT_DIR / experiment_name / "data"
    with servers(launch_engine, output_dir, gpu) as server_dir:
        write_decode_file(build_decode_requests(decode_lengths[0], 1, request_count), gpu)
        return _serve_run(
            output_dir,
            server_dir,
            decode_lengths,
            prefill_lengths,
            request_count,
            arrival_rate,
            decode_ratio,
            served_kinds,
            gpu,
        )


def sweep(
    experiment_name="sweep",
    decode_ratios=DECODE_RATIOS,
    run_solos=True,
    decode_lengths=DECODE_LENGTHS,
    prefill_lengths=PREFILL_LENGTHS,
    request_count=REQUEST_COUNT,
    arrival_rate=ARRIVAL_RATE,
    gpu=GPU,
    launch_engine=True,
):
    """Run every decode ratio into OUTPUT_DIR/experiment_name/data/<ratio>/ on one launch of the servers, or those of launch_interference.py without launch_engine, after a discarded warm-up run: a mixed run, plus each kind served alone at the ratios between 0 and 1 as its baseline without interference."""
    sweep_dir = OUTPUT_DIR / experiment_name / "data"
    _log(f"Sweep into {sweep_dir}")
    with servers(launch_engine, sweep_dir, gpu) as server_dir:
        write_decode_file(build_decode_requests(decode_lengths[0], 1, max(request_count, WARMUP_REQUEST_COUNT)), gpu)
        _log(f"[warm-up] decode_ratio=0.5 mixed, {WARMUP_REQUEST_COUNT} requests")
        with log_indent():
            _serve_run(
                sweep_dir / "warmup",
                server_dir,
                decode_lengths,
                prefill_lengths,
                WARMUP_REQUEST_COUNT,
                arrival_rate,
                0.5,
                ("decode", "prefill"),
                gpu,
            )
        for index, decode_ratio in enumerate(decode_ratios, 1):
            served_by_run_kind = {"mixed": ("decode", "prefill")}
            if run_solos and 0 < decode_ratio < 1:
                served_by_run_kind |= {"decode_only": ("decode",), "prefill_only": ("prefill",)}
            for run_kind, served_kinds in served_by_run_kind.items():
                _log(f"[{index}/{len(decode_ratios)}] decode_ratio={decode_ratio:g} {run_kind}")
                with log_indent():
                    _serve_run(
                        sweep_dir / f"{decode_ratio:g}" / run_kind,
                        server_dir,
                        decode_lengths,
                        prefill_lengths,
                        request_count,
                        arrival_rate,
                        decode_ratio,
                        served_kinds,
                        gpu,
                    )


if __name__ == "__main__":
    # run(experiment_name, decode_lengths, prefill_lengths, request_count, arrival_rate, decode_ratio, served_kinds, gpu, launch_engine)
    # - experiment_name: the run's data goes to OUTPUT_DIR/experiment_name/data (full server logs in data/servers, or SERVER_DIR without launch_engine), its plots to OUTPUT_DIR/experiment_name/plot
    # - decode_lengths: (prompt_length, output_length) of decode requests; the KV of the first request_count decode prompts is written to KV_DIR before the run and deleted after it
    # - prefill_lengths: (prompt_length, output_length) of prefill requests; output_length 1 makes them pure prefills
    # - request_count: number of request slots, each filled by a decode or a prefill request
    # - arrival_rate: requests per second, at seeded Poisson send times; None sends every request at time 0
    # - decode_ratio: fraction of slots that are decode, 0 to 1
    # - served_kinds: kinds actually sent; ("decode", "prefill") by default, ("decode",) or ("prefill",) for a solo run
    # - gpu: GPU id, GPU by default
    # - launch_engine: launch fresh servers for this call if True (default), else use the running ones of launch_interference.py; either way each run starts from an idle engine and an empty LMCache L1
    run("smoke_test2", DECODE_LENGTHS, PREFILL_LENGTHS, 16, ARRIVAL_RATE, 0.5, launch_engine=False)
    # sweep(launch_engine=False)
