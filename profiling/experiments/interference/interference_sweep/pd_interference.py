"""Interference between the prefill and decode of the same prompts on one engine, served colocated as one request each, then disaggregated as every prompt's prefill into the decode files followed by every decode from them.

Run from this directory on a GPU node with the conserve env active. run launches its own servers; with launch_engine=False it uses those of python launch_interference.py running in another terminal on the same node:
    python pd_interference.py
    python -c "from pd_interference import *; run_single('test', LENGTHS, 16, ARRIVAL_RATE)"
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import GPU, KV_DIR, OUTPUT_ROOT, _log, base_properties, check, delete_decode_file, engine_logs, l1_request_cap, load_decode_file, log_indent, send, send_times, servers, wipe_engine, write_decode_file, write_properties
from metric import parse_engine_logs
from token_generation import load_standard

LENGTHS = (8192, 512)
REQUEST_COUNT = 32
ARRIVAL_RATE = 2.0
PROMPT_SEED = 0
OUTPUT_DIR = OUTPUT_ROOT / Path(__file__).stem


# Requests


def build_requests(request_kind, prompt_length, output_length, request_count, arrival_rate):
    """request_kind requests on the first request_count prompts of prompt_length, each generating output_length tokens, with Poisson send times at arrival_rate (all at time 0 if arrival_rate is None)."""
    return [
        {"id": f"{request_kind}-{index:04d}", "request_kind": request_kind, "prompt": prompt_record["token_ids"], "max_tokens": output_length, "send_time": float(send_time), "priority": index}
        for index, (prompt_record, send_time) in enumerate(zip(load_standard({prompt_length: request_count}, PROMPT_SEED), send_times(arrival_rate, request_count)))
    ]


# Run


def _serve_decodes(timed_requests, lengths, gpu):
    """Serve decode timed_requests in groups that fit in L1, emptying L1 and loading each group's KV into it first; return their results and the chunks found and expected."""
    group_size = l1_request_cap(*lengths)
    request_results = []
    chunk_counts = {"found_chunks": 0, "total_chunks": 0}

    wipe_engine(gpu)
    
    for start in range(0, len(timed_requests), group_size):
        group = timed_requests[start : start + group_size]
        if start != 0:
            wipe_engine(gpu, wipe_vllm=False)
        _log(f"Loading KV of {len(group)} decode prompts into L1...")
        found_chunks, total_chunks = load_decode_file(group)
        _log(f"Loaded {found_chunks}/{total_chunks} chunks from file")
        if found_chunks < total_chunks:
            raise RuntimeError(f"Only {found_chunks}/{total_chunks} decode chunks found in {KV_DIR}")
        chunk_counts["found_chunks"] += found_chunks
        chunk_counts["total_chunks"] += total_chunks
        request_results += asyncio.run(send(group, gpu))
    return request_results, chunk_counts


def _serve_run(
    output_dir,
    server_dir,
    request_kind,
    lengths,
    request_count,
    arrival_rate,
    max_num_seqs,
    gpu,
):
    """One run of request_kind requests on servers already running with logs in server_dir: wipe them and serve the requests, prefills writing their KV to the decode files and decodes loading it from them in groups that fit in L1, then save this run's slice of the engine logs, the per-request timings and run.properties to output_dir."""
    timed_requests = build_requests(request_kind, lengths[0], 1 if request_kind == "prefill" else lengths[1], request_count, arrival_rate)
    chunk_counts = {}
    wipe_engine(gpu)
    with engine_logs(server_dir, output_dir, gpu):
        if request_kind == "prefill":
            request_results = write_decode_file(timed_requests, gpu)
        elif request_kind == "decode":
            request_results, chunk_counts = _serve_decodes(timed_requests, lengths, gpu)
        else:
            request_results = asyncio.run(send(timed_requests, gpu))
    parse_engine_logs(request_results, {gpu: "kv_both"}, output_dir)
    write_properties(
        output_dir / "run.properties",
        {
            "request_kind": request_kind,
            "prompt_length": lengths[0],
            "output_length": lengths[1],
            "request_count": request_count,
            "arrival_rate": arrival_rate,
            "prompt_seed": PROMPT_SEED,
            **chunk_counts,
            **check(request_results, (request_kind,)),
            **base_properties(gpu, server_dir, max_num_seqs),
        },
    )
    return request_results


def _serve_runs(
    data_dir,
    server_dir,
    lengths,
    request_count,
    arrival_rate,
    max_num_seqs,
    gpu,
):
    """Serve the same requests colocated, then disaggregated as every prefill followed by every decode, into data_dir on servers already running with logs in server_dir, then delete the decode files."""
    runs = (
        ("colocated", "colocated", request_count),
        ("disaggregated/prefill", "prefill", request_count),
        ("disaggregated/decode", "decode", request_count),
    )
    for run_name, request_kind, run_request_count in runs:
        _log(f"[{run_name}] {run_request_count} {request_kind} requests")
        with log_indent():
            _serve_run(data_dir / run_name, server_dir, request_kind, lengths, run_request_count, arrival_rate, max_num_seqs, gpu)
    delete_decode_file()
    KV_DIR.chmod(0o550)


def run_single(
    experiment_name,
    lengths,
    request_count,
    arrival_rate=0,
    max_num_seqs=None,
    gpu=GPU,
    launch_engine=True,
):
    """Serve the same requests colocated, then disaggregated as every prefill followed by every decode, into OUTPUT_DIR/experiment_name/data/ on one launch of the servers capped at max_num_seqs requests per step if given, or those of launch_interference.py without launch_engine."""
    data_dir = OUTPUT_DIR / experiment_name / "data"
    _log(f"Runs into {data_dir}")
    with servers(launch_engine, data_dir, gpu, max_num_seqs) as server_dir:
        _serve_runs(data_dir, server_dir, lengths, request_count, arrival_rate, max_num_seqs, gpu)


def run_sweep(
    experiment_name,
    input_length_list,
    output_length_list,
    request_count,
    arrival_rate=0,
    max_num_seqs_list=(None,),
    gpu=GPU,
):
    """Run every input_length and output_length pair into OUTPUT_DIR/experiment_name/max_num_seqs/input_length/output_length/data/ on one launch of the servers per max_num_seqs."""
    for max_num_seqs in max_num_seqs_list:
        sweep_dir = OUTPUT_DIR / experiment_name / str(max_num_seqs)
        _log(f"max_num_seqs={max_num_seqs}")
        with log_indent(), servers(True, sweep_dir, gpu, max_num_seqs) as server_dir:
            for input_length in input_length_list:
                for output_length in output_length_list:
                    _log(f"input_length={input_length}, output_length={output_length}")
                    with log_indent():
                        _serve_runs(
                            sweep_dir / str(input_length) / str(output_length) / "data",
                            server_dir,
                            (input_length, output_length),
                            request_count,
                            arrival_rate,
                            max_num_seqs,
                            gpu,
                        )


if __name__ == "__main__":
    # run(experiment_name, lengths, request_count, arrival_rate, max_num_seqs, gpu, launch_engine)
    # - experiment_name: the runs' data goes to OUTPUT_DIR/experiment_name/data/{colocated,disaggregated/prefill,disaggregated/decode} (full server logs in data/servers, or SERVER_DIR without launch_engine)
    # - lengths: (prompt_length, output_length) of every request; disaggregated, each prefill generates 1 token while its KV is written to the decode files, and each decode output_length tokens from them
    # - request_count: number of requests, on the first request_count prompts of the token stream
    # - arrival_rate: requests per second, at seeded Poisson send times shared by every run; None sends every request at time 0
    # - max_num_seqs: cap on requests per engine step (vLLM --max-num-seqs), needs launch_engine; None keeps vLLM's default, 256 on the A40
    # - gpu: GPU id, GPU by default
    # - launch_engine: launch fresh servers for this call if True (default), else use the running ones of launch_interference.py; either way each run starts from an idle engine and an empty LMCache L1
    run_single("test2", (16384, 512), 32, launch_engine=True)
    # run_sweep("test_ordering2", [1024, 2048, 4096], [512, 1024], 16)
    # run_sweep("length_sweep_ordered", [16384, 32768], [128, 256, 512, 1024, 2048], 32)
    # print(build_requests("test", 1024, 512, 16, 0))
