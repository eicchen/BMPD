"""Shared LMCache MP server and per-GPU vLLM engines: launch and the request path."""

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import openai.resources.completions  # noqa: F401 -- loads otherwise on the first request, stalling it ~0.9 s
import requests
from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "config"))
from config import GPU_MEM_UTIL, MAX_MODEL_LEN, MODEL, MODEL_DIR, PROFILE

ENGINE_PORT = 7100
LMCACHE_PORT, LMCACHE_HTTP_PORT = 7600, 7601
TELEMETRY_PORT = 7602
L1_SIZE_GB = 60
CHUNK_SIZE = 256
MARKER = "conserve_pd"

progress = logging.getLogger("progress")


def _spawn(cmd, env, log):
    """Start cmd with env added and its output in log; return the process."""
    with open(log, "w") as f:
        return subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**os.environ, **env},
        )


def _wait(proc, url, what):
    """Return once url answers; raise if proc exits first."""
    while proc.poll() is None:
        try:
            if requests.get(url).ok:
                return
        except requests.ConnectionError:
            pass
        time.sleep(5)
    raise RuntimeError(f"{what} exited with {proc.returncode}")


def launch_lmcache(gpus, log_dir, l2_path=None):
    """Start the LMCache MP server shared by the engines on gpus, with a filesystem L2 store at l2_path if given, logging into log_dir; return it once healthy."""
    cmd = [
        "lmcache",
        "server",
        "--host",
        "localhost",
        "--port",
        str(LMCACHE_PORT),
        "--http-host",
        "localhost",
        "--http-port",
        str(LMCACHE_HTTP_PORT),
        "--l1-size-gb",
        str(L1_SIZE_GB),
        "--eviction-policy",
        "LRU",
        "--chunk-size",
        str(CHUNK_SIZE),
        "--max-gpu-workers",
        str(len(gpus)),
        "--max-cpu-workers",
        "2",
    ]
    if l2_path:
        cmd += ["--l2-adapter", json.dumps({"type": "fs", "base_path": str(l2_path)})]
    env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
    log = log_dir / "lmcache.log"
    proc = _spawn(cmd, env, log)
    _wait(proc, f"http://localhost:{LMCACHE_HTTP_PORT}/healthcheck", f"lmcache server, see {log},")
    return proc


def launch_engine(gpu, role, log_dir, connector=None):
    """Start a kv_producer or kv_consumer engine with KV connector, or a colocated one, on gpu serving ENGINE_PORT + gpu, logging into log_dir; return it without waiting."""
    port = ENGINE_PORT + gpu
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "engine.jsonl").unlink(missing_ok=True)
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--port",
        str(port),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEM_UTIL),
        "--enable-prompt-tokens-details",
        "--enable-logging-iteration-details",
        "--core-log-file",
        str(log_dir / "core.jsonl"),
        "--engine-log-file",
        str(log_dir / "engine.jsonl"),
        *(["--no-enable-prefix-caching"] if connector == "LMCacheMPConnector" else []),
        *PROFILE.get("vllm_serve_flags", []),
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HF_HUB_CACHE": str(MODEL_DIR),
    }
    if role != "colocated":
        kv = {
            "kv_connector": connector,
            "kv_role": "kv_both" if connector == "LMCacheAdaptiveConnector" else role,
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": LMCACHE_PORT,
            },
        }
        if connector == "LMCacheAdaptiveConnector":
            kv["kv_connector_module_path"] = "lmcache_adaptive_connector"
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(HERE), os.environ.get("PYTHONPATH"))))
        if connector == "LMCacheMPConnector" and role == "kv_producer":
            env |= {
                "LMCACHE_REQUEST_TELEMETRY_TYPE": "fastapi",
                "LMCACHE_REQUEST_TELEMETRY_ENDPOINT": f"http://localhost:{TELEMETRY_PORT}/api/v1/telemetry",
            }
        cmd += ["--kv-transfer-config", json.dumps(kv)]
    return _spawn(cmd, env, log_dir / "server.log")


def wait_engine(proc, gpu):
    """Return once the engine on gpu is healthy; raise if it exits first."""
    _wait(proc, f"http://localhost:{ENGINE_PORT + gpu}/health", f"vLLM on GPU {gpu}, see gpu{gpu}/server.log,")


def kv_capacity(gpu):
    """KV cache capacity in tokens of the engine on gpu, from its metrics."""
    info = re.search(r"vllm:cache_config_info\{([^}]*)\}", requests.get(f"http://localhost:{ENGINE_PORT + gpu}/metrics").text).group(1)
    labels = dict(re.findall(r'(\w+)="([^"]*)"', info))
    return int(labels["num_gpu_blocks"]) * int(labels["block_size"])


def client(gpu):
    """OpenAI client for the engine on gpu."""
    return AsyncOpenAI(
        base_url=f"http://localhost:{ENGINE_PORT + gpu}/v1",
        api_key="EMPTY",
        max_retries=0,
    )


async def complete(client, rec, max_tokens=None, params=None, token_ids=False, handoff=None):
    """Stream rec from an engine with connector params if given, ending at the first chunk after the future handoff() returns at its first token is done; return its timings, token counts and, if token_ids, its prompt and output token ids and output text."""
    extra_body = {"request_id": rec["id"], "ignore_eos": True, "prompt": rec["prompt"]}
    if params:
        extra_body["kv_transfer_params"] = {MARKER: params}
    if token_ids:
        extra_body["return_token_ids"] = True
    sent = time.time()
    stream = await client.completions.create(
        model=MODEL,
        prompt="",
        max_tokens=max_tokens or rec["max_tokens"],
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra_body,
    )
    first = usage = stop = None
    prompt_ids, out_ids, text = [], [], ""
    async for chunk in stream:
        if first is None and chunk.choices:
            first = time.time()
            stop = handoff() if handoff else None
        if token_ids:
            for c in chunk.choices:
                prompt_ids += c.prompt_token_ids or []
                out_ids += c.token_ids
                text += c.text
        if chunk.usage:
            usage = chunk.usage
        if stop is not None and stop.done():
            await stream.close()
            break
    details = usage.prompt_tokens_details if usage else None
    return {
        "sent": sent,
        "first_token": first,
        "done": time.time(),
        "prompt_tokens": usage.prompt_tokens if usage else len(prompt_ids),
        "completion_tokens": usage.completion_tokens if usage else len(out_ids),
        "cached_tokens": details.cached_tokens if details else 0,
        **({"token_ids": prompt_ids + out_ids, "text": text} if token_ids else {}),
    }


async def converse(turns, serve):
    """Serve a conversation's turns in order with serve(rec) returning (row, token ids, output text), each prompt the previous one and its output plus the turn's own, as text or token ids like the records, with its token count, estimated from the trace for text, as "prompt_size"; return their rows."""
    as_text = isinstance(turns[0]["prompt"], str)
    rows, history, ids = [], "" if as_text else [], []
    for rec in turns:
        prompt = history + (rec["prompt"] if as_text else rec["ids"])
        size = len(ids) + (rec["in_token_size"] if as_text else len(rec["ids"]))
        row, ids, out = await serve({**rec, "prompt": prompt, "prompt_size": size})
        history = prompt + out if as_text else ids
        rows.append(row)
    progress.info(
        "conv %s done: %d turns in %.1f s",
        turns[0]["conv"],
        len(rows),
        rows[-1]["gpu_requests"][-1]["done"] - rows[0]["gpu_requests"][0]["sent"],
    )
    return rows
