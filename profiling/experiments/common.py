"""Shared vLLM launch, decode load and prefill admission for the interference experiments."""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import requests
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "config"), str(ROOT / "profiling" / "scripts")]
from config import GPU_TYPE, MAX_MODEL_LEN, MODEL, MODEL_DIR, PROFILE, TENSOR_PARALLEL_SIZE
from token_generation import LOAD_SEED, load

OUTPUT_DIR = ROOT / "profiling" / "output"
PORT = 8000
URL = f"http://localhost:{PORT}"
KV_CACHE_BYTES = 20 * 2**30
ARRIVAL_SEED = 0
LOGS = ("server.log", "core.jsonl")


def launch_vllm(log_dir, flags):
    """Start vllm serve with the given extra flags logging into log_dir; return the process once it is healthy."""
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--port",
        str(PORT),
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--kv-cache-memory-bytes",
        str(KV_CACHE_BYTES),
        "--no-async-scheduling",
        "--enable-logging-iteration-details",
        "--core-log-file",
        str(log_dir / "core.jsonl"),
        *flags,
        *PROFILE.get("vllm_serve_flags", []),
    ]
    with open(log_dir / "server.log", "w") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_CACHE": str(MODEL_DIR)},
        )
    while proc.poll() is None:
        try:
            if requests.get(f"{URL}/health").ok:
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(5)
    raise RuntimeError(f"vllm exited with {proc.returncode}, see {log_dir / 'server.log'}")


def check_vllm():
    requests.get(f"{URL}/health").raise_for_status()


async def _start_decode(client, rec, decode_output):
    """Open one decode stream of decode_output tokens, never-ending if None; return it and its send time after its first token."""
    sent = time.time()
    stream = await client.completions.create(
        model=MODEL,
        prompt=rec["token_ids"],
        max_tokens=decode_output or MAX_MODEL_LEN - rec["length"],
        stream=True,
        extra_body={"ignore_eos": True, "request_id": rec["id"]},
    )
    await anext(stream)
    return stream, sent


async def _drain(rec, stream, sent):
    """Read a decode stream to its end; return its send and finish times."""
    async with stream:
        async for _ in stream:
            pass
    return {"id": rec["id"], "role": "decode", "sent": sent, "done": time.time()}


async def load_decode(client, decode_count, decode_length, decode_output):
    """Start decode_count decodes of decode_length tokens; return their drain tasks once all are decoding."""
    recs = load({decode_length: decode_count}, {})
    started = await asyncio.gather(*(_start_decode(client, r, decode_output) for r in recs))
    return [asyncio.create_task(_drain(r, *s)) for r, s in zip(recs, started)]


async def _prefill(client, rec):
    """Send one max_tokens=1 prefill; return its send and finish times and the server's prompt token count."""
    sent = time.time()
    resp = await client.completions.create(
        model=MODEL,
        prompt=rec["token_ids"],
        max_tokens=1,
        extra_body={"request_id": rec["id"]},
    )
    done = time.time()
    return {
        "id": rec["id"],
        "role": "prefill",
        "sent": sent,
        "done": done,
        "prompt_tokens": resp.usage.prompt_tokens,
    }


async def admit_prefill(client, prefill_length, arrival_rate, prefill_count):
    """Send prefill_count prefills of prefill_length at seeded Poisson arrivals; return their timings."""
    recs = load({}, {prefill_length: prefill_count})
    gaps = np.random.default_rng(ARRIVAL_SEED).exponential(1 / arrival_rate, prefill_count)
    tasks = []
    for rec, gap in zip(recs, gaps):
        await asyncio.sleep(gap)
        tasks.append(asyncio.create_task(_prefill(client, rec)))
    return await asyncio.gather(*tasks)


async def _measure(log_dir, prefill, decode):
    """Run the given workloads, cancelling never-ending decodes after the prefills; return request timings and each log's byte span from the decodes starting."""
    async with AsyncOpenAI(
        base_url=f"{URL}/v1",
        api_key="EMPTY",
        max_retries=0,
    ) as client:
        decodes = await load_decode(client, **decode) if decode else []
        start = [(log_dir / n).stat().st_size for n in LOGS]
        reqs = await admit_prefill(client, **prefill) if prefill else []
        if decode.get("decode_output"):
            reqs += await asyncio.gather(*decodes)
        else:
            for t in decodes:
                t.cancel()
            await asyncio.gather(*decodes, return_exceptions=True)
        end = [(log_dir / n).stat().st_size for n in LOGS]
    return reqs, zip(LOGS, start, end)


def _run(out, log_dir, prefill, decode):
    """Run the given workloads on the running server and write their outputs to out."""
    check_vllm()
    out.mkdir(parents=True, exist_ok=True)
    reqs, spans = asyncio.run(_measure(log_dir, prefill, decode))
    for name, start, end in spans:
        with open(log_dir / name, "rb") as f:
            f.seek(start)
            (out / name).write_bytes(f.read(end - start))
    (out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    props = {
        **prefill,
        **decode,
        "arrival_seed": ARRIVAL_SEED,
        "load_seed": LOAD_SEED,
        "max_model_len": MAX_MODEL_LEN,
        "kv_cache_bytes": KV_CACHE_BYTES,
        "model": MODEL,
        "gpu_type": GPU_TYPE,
        "vllm_version": version("vllm"),
        "openai_version": version("openai"),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "run.properties").write_text("".join(f"{k}={v}\n" for k, v in props.items()))


def run_decode_only(log_dir, decode_count, decode_length, decode_output):
    """Run decode_count decodes alone on the running server and write their outputs."""
    decode = dict(decode_count=decode_count, decode_length=decode_length, decode_output=decode_output)
    _run(log_dir / "decode_only", log_dir, {}, decode)


def run_prefill_only(log_dir, prefill_length, arrival_rate, prefill_count):
    """Run prefill_count prefills alone on the running server and write their outputs."""
    prefill = dict(prefill_length=prefill_length, arrival_rate=arrival_rate, prefill_count=prefill_count)
    out = log_dir / "prefill_only" / str(prefill_length) / f"{arrival_rate:g}"
    _run(out, log_dir, prefill, {})


def run_point(log_dir, prefill_length, arrival_rate, prefill_count, decode_count, decode_length, decode_output):
    """Run prefills over decodes on the running server and write their outputs."""
    prefill = dict(prefill_length=prefill_length, arrival_rate=arrival_rate, prefill_count=prefill_count)
    decode = dict(decode_count=decode_count, decode_length=decode_length, decode_output=decode_output)
    out = log_dir / "colocated" / str(prefill_length) / f"{arrival_rate:g}"
    _run(out, log_dir, prefill, decode)
