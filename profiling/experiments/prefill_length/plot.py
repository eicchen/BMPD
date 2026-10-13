"""Plot the prefill length experiment.

Run from this directory with the conserve env active:
    python plot.py
    python -c "from plot import plot; plot(2.0)"
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUTPUT_DIR
from prefill_length import ARRIVAL_RATE, EXPERIMENT

ITERATION = re.compile(
    r"Iteration\(\d+\): (\d+) context requests, (\d+) context tokens, "
    r"(\d+) generation requests, (\d+) generation tokens, iteration elapsed time: ([\d.]+) ms",
)
COLORS = {"chunked": "#2a78d6", "no_chunked": "#eb6834"}
SURFACE, INK, SECONDARY, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MARK = dict(lw=2, marker="o", ms=8, mec=SURFACE, mew=2)


def _requests(point):
    """Request records in one point."""
    return [json.loads(l) for l in (point / "requests.jsonl").read_text().splitlines()]


def prefill_latency(point):
    """Median prefill latency (ms) in one point."""
    return np.median([r["done"] - r["sent"] for r in _requests(point) if r["role"] == "prefill"]) * 1000


def e2e_time(point):
    """Seconds from the first request sent to the last request finished in one point."""
    reqs = _requests(point)
    return max(r["done"] for r in reqs) - min(r["sent"] for r in reqs)


def read_point(point):
    """Window step time, decode-only step time (ms), prefill latency (ms), e2e (s), and decode count, length and output for one colocated point."""
    props = dict(l.split("=", 1) for l in (point / "run.properties").read_text().splitlines())
    steps = np.array(ITERATION.findall((point / "server.log").read_text()), dtype=float)
    _, ctx_tokens, gen_requests, _, ms = steps.T
    start = np.argmax(gen_requests == int(props["decode_count"]))
    end = len(ctx_tokens) - np.argmax(ctx_tokens[::-1] > 0)
    ctx, ms = ctx_tokens[start:end], ms[start:end]
    return (
        ms.mean(),
        ms[ctx == 0].mean(),
        prefill_latency(point),
        e2e_time(point),
        int(props["decode_count"]),
        int(props["decode_length"]),
        int(props["decode_output"]),
    )


def _line(ax, x, y, color, label):
    """Plot one series and label its last point."""
    ax.plot(x, y, color=color, label=label, **MARK)
    ax.annotate(label, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=SECONDARY)


def _style(ax, title, lengths):
    """Apply recessive axes, a log2 x axis ticked at lengths, a log y axis and a left-aligned title."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(lengths, [str(l) for l in lengths])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Prefill length (tokens)", color=SECONDARY)
    ax.margins(x=0.25)
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)


def plot(arrival_rate):
    """Plot TBT, prefill latency and e2e against prefill length for chunked and no_chunked, with solo baselines."""
    exp = OUTPUT_DIR / EXPERIMENT
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor=SURFACE)
    all_lengths, decodes = set(), set()
    for mode, color in COLORS.items():
        data = exp / "data" / mode
        points = sorted((data / "colocated").glob(f"*/{arrival_rate:g}"), key=lambda p: int(p.parent.name))
        if not points:
            continue
        lengths = [int(p.parent.name) for p in points]
        all_lengths.update(lengths)
        step_ms, decode_only_ms, latency, e2e, *decode = np.array([read_point(p) for p in points]).T
        decodes.update(zip(*(d.astype(int) for d in decode)))
        solo = [data / "prefill_only" / p.relative_to(data / "colocated") for p in points]
        _line(axes[0], lengths, step_ms, color, mode)
        axes[0].plot(lengths, decode_only_ms, color=color, alpha=0.4, label=f"{mode} decode-only", **MARK)
        _line(axes[1], lengths, latency, color, mode)
        axes[1].plot(lengths, [prefill_latency(p) for p in solo], color=color, alpha=0.4, label=f"{mode} prefill-only", **MARK)
        solo_decode = e2e_time(data / "decode_only")
        _line(axes[2], lengths, e2e, color, mode)
        axes[2].axhline(solo_decode, color=color, alpha=0.4, lw=2, label=f"{mode} decode-only")
        axes[2].plot(lengths, [e2e_time(p) + solo_decode for p in solo], color=color, alpha=0.4, label=f"{mode} prefill-only + decode-only", **MARK)
    for ax, title in zip(axes, ("Mean TBT (ms)","Median prefill latency (ms)", "E2E time (s)")):
        _style(ax, title, sorted(all_lengths))
        ax.legend(frameon=False, labelcolor=SECONDARY)
    axes[2].yaxis.set_minor_formatter(plt.FormatStrFormatter("%g"))
    decode = ", ".join(f"{c} decodes of {l}-token prompts × {o} output tokens" for c, l, o in sorted(decodes))
    fig.suptitle(f"Prefill length sweep at {arrival_rate:g} req/s over {decode}", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    (exp / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(exp / "plots" / "plot.png", dpi=150)


if __name__ == "__main__":
    plot(ARRIVAL_RATE)
