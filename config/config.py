"""Model, GPU and dataset settings from config.env, model_specific_configs.toml and the node's Slurm gres."""

import re
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import tomli

CONFIG_DIR = Path(__file__).resolve().parent


def _env():
    """config.env as a dict; a later line overrides an earlier one."""
    lines = (l.strip() for l in (CONFIG_DIR / "config.env").read_text().splitlines())
    pairs = (l.split("=", 1) for l in lines if "=" in l and not l.startswith("#"))
    return {k.strip(): v.strip() for k, v in pairs}


def _gpu_type():
    """This node's GPU model from its Slurm gres, e.g. L40S; None on a node without GPUs."""
    gres = subprocess.run(
        ["sinfo", "-h", "-n", socket.gethostname(), "-o", "%G"],
        capture_output=True,
        text=True,
    ).stdout
    m = re.search(r"gpu:([^:]+):", gres)
    return m and m[1].upper()


_e = _env()
MODEL = _e["MODEL"]
MODEL_DIR = Path(_e["MODEL_DIR"])
DATASET_DIR = Path(_e["DATASET_DIR"])
GPU_TYPE = _gpu_type()

with open(CONFIG_DIR / "model_specific_configs.toml", "rb") as f:
    PROFILE = tomli.load(f)["models"][MODEL]
TENSOR_PARALLEL_SIZE = PROFILE.get("tensor_parallel_size", 1)
GPU_MEM_UTIL = PROFILE.get("gpu_mem_util", 0.92)
MAX_MODEL_LEN = PROFILE["max_model_len"]

SH_VARS = (
    "MODEL",
    "MODEL_DIR",
    "DATASET_DIR",
    "GPU_TYPE",
    "TENSOR_PARALLEL_SIZE",
    "GPU_MEM_UTIL",
)

if __name__ == "__main__" and sys.argv[1:] == ["--sh-vars"]:
    for k in SH_VARS:
        print(f"export {k}={shlex.quote(str(globals()[k]))}")
    flags = " ".join(shlex.quote(f) for f in PROFILE.get("vllm_serve_flags", []))
    print(f"VLLM_SERVE_FLAGS=({flags})")
