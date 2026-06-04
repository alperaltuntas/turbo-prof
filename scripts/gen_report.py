#!/usr/bin/env python3
"""Generate a Markdown performance report from MOM6 double_gyre scaling runs.

The companion sweep script (run-scaling-sweep.sh cpu|gpu) sweeps a job-size index
`i` and leaves one `<platform>_<i>.out` log per run in the run directory. Each
log ends with an FMS mpp_clock table; we read the cross-PE mean ("tavg") of the
"Main loop" timer as the per-run measurement.

The methodology fixes the number of *dynamic* steps at 150 for every run
(TIMEUNIT = dt together with DAYMAX = 150), so wall-clock is directly
comparable across problem sizes. See docs/METHODOLOGY.md for the full rationale.

This is the "scaling" report type. Reusable building blocks (log parsing,
provenance capture) live in the `turboprof` package so future report types can
share them; only the scaling-specific plots, tables, and prose live here.

Usage (under an env with matplotlib, e.g. conda `npl`):

    python3 gen_report.py --cpu-dir DIR [--gpu-dir DIR] \
        [--stack-dir DIR] [--outdir DIR] [--nsteps 150] [--title "..."]

GPU is optional: if --gpu-dir is omitted (or empty), the GPU sections render as
"pending" and the same command can be re-run later to fill them in.
"""

import argparse
import csv
import datetime
import json
import os

from turboprof.parsing import (
    NK, BLOCK, CPU_PER_NODE, NSTEPS_DEFAULT, collect, add_throughput)
from turboprof.provenance import gather_provenance, render_provenance, render_stamp

# Reference operating points the team cares about, in gridpoints per GPU.
REF_POINTS = [
    (720 * 360 * 75, "720x360x75 = 19.4M (no MARBL)"),
    (360 * 360 * 75, "360x360x75 = 9.7M (MARBL)"),
]

# The continuity solver's mpp_clock name (see parsing.py).
CONTINUITY_TIMER = "(Ocean continuity equation)"

# The barotropic solver's top-level mpp_clock; its sub-steps share the
# "(Ocean BT " name prefix (pre-calcs, halo updates, stepping, post-calcs).
BAROTROPIC_TIMER = "(Ocean barotropic mode stepping)"
BAROTROPIC_PREFIX = "(Ocean BT "

# Job-size index used for the per-routine breakdown: i=128 is the first clean
# 1-CPU-node-vs-1-GPU point (13.1M gridpoints, nearest to the 19.4M production
# size). Overridable with --breakdown-i.
BREAKDOWN_I_DEFAULT = 128


# --- plots ------------------------------------------------------------------

def plot_cpu_timing(cpu_rows, outpath):
    """Single panel: main-loop time vs problem size across both CPU regimes.

    The sweep is one continuous problem-size scan: up to the node cap both
    gridpoints and ranks grow together (weak scaling); beyond it only the
    problem grows while ranks stay at 128 (saturated node). A vertical divider
    marks the boundary.
    """
    import matplotlib.pyplot as plt
    if not cpu_rows:
        return None

    x = [r["gridpoints"] for r in cpu_rows]
    t = [r["main_loop"] for r in cpu_rows]
    boundary = BLOCK * BLOCK * NK * CPU_PER_NODE  # gridpoints at i = 128

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, t, "o-", color="C0", label="measured")
    ax.axvline(boundary, ls=":", color="red", alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    _, hi = ax.get_ylim()
    ax.text(boundary * 0.85, hi * 0.7, "ranks grow\n(weak scaling)",
            ha="right", va="top", fontsize=9, color="dimgray")
    ax.text(boundary * 1.15, hi * 0.7, "128 ranks fixed\n(work/rank grows)",
            ha="left", va="top", fontsize=9, color="dimgray")

    ax.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    ax.set_ylabel("Main loop wall time (s)")
    ax.set_title("CPU timing vs problem size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


def plot_throughput(cpu_rows, gpu_rows, outpath):
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    fig, ax = plt.subplots(figsize=(8, 5))

    if cpu_rows:
        x = [r["gridpoints"] for r in cpu_rows]
        y = [r["throughput"] for r in cpu_rows]
        ax.plot(x, y, "s-", label="CPU node (<=128 ranks)", color="C0")
    if gpu_rows:
        x = [r["gridpoints"] for r in gpu_rows]
        y = [r["throughput"] for r in gpu_rows]
        ax.plot(x, y, "o-", label="1 GPU (A100)", color="C1")

    # x in data coords, y in axes-fraction so labels sit just above the x-axis
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for gp, label in REF_POINTS:
        ax.axvline(gp, ls="--", alpha=0.6, color="gray")
        ax.text(gp, 0.02, " " + label, rotation=90, transform=trans,
                va="bottom", ha="right", fontsize=8, color="gray")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    ax.set_ylabel("Throughput (cell-updates / s)")
    ax.set_title("Throughput vs problem size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


def plot_speedup(cpu_rows, gpu_rows, outpath):
    """GPU-to-CPU throughput ratio vs problem size -- the crossover plot.

    Speedup = single-GPU throughput / CPU-node throughput, matched by job-size
    index `i` (identical problem size). A solid line at 1.0 marks parity; above
    it the GPU wins, below it the full CPU node wins. The shaded band (problem
    >= the i=128 node-saturation point) is where the comparison is a clean
    1-GPU-vs-1-full-node match per METHODOLOGY.md; left of it the CPU branch is
    still weak-scaling on fewer than 128 ranks, so the ratio mixes regimes.
    """
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    if not cpu_rows or not gpu_rows:
        return None

    cpu_by_i = {r["i"]: r for r in cpu_rows}
    pts = sorted((g["gridpoints"], g["throughput"] / cpu_by_i[g["i"]]["throughput"])
                 for g in gpu_rows if g["i"] in cpu_by_i)
    if not pts:
        return None
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    boundary = BLOCK * BLOCK * NK * CPU_PER_NODE  # gridpoints at i = 128

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "o-", color="C2", label="GPU / CPU-node throughput")
    ax.axhline(1.0, ls="-", color="black", alpha=0.6, lw=1)

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    if max(x) >= boundary:
        ax.axvspan(boundary, max(x), color="C0", alpha=0.08)
        ax.text(boundary, 0.97, " 1 node vs 1 GPU\n (clean comparison)",
                transform=trans, va="top", ha="left", fontsize=8, color="dimgray")
    for gp, label in REF_POINTS:
        ax.axvline(gp, ls="--", alpha=0.6, color="gray")
        ax.text(gp, 0.02, " " + label, rotation=90, transform=trans,
                va="bottom", ha="right", fontsize=8, color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    ax.set_ylabel("Speedup (GPU throughput / CPU-node throughput)")
    ax.set_title("GPU-vs-CPU speedup vs problem size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


def plot_init(cpu_rows, gpu_rows, outpath):
    """Initialization wall time vs problem size, CPU node vs GPU.

    `init` is the FMS "Initialization" timer -- setup before the main loop. On
    the GPU it grows steeply (device allocation + host->device staging + kernel
    setup), a fixed per-run cost the main-loop throughput numbers don't capture;
    on the CPU node it stays modest. Surfacing it explains part of the GPU's
    disadvantage and its growth toward the memory ceiling at large sizes.
    """
    import matplotlib.pyplot as plt
    series = []
    if cpu_rows:
        series.append(("s-", "C0", "CPU node (<=128 ranks)", cpu_rows))
    if gpu_rows:
        series.append(("o-", "C1", "1 GPU (A100)", gpu_rows))
    if not any(any(r.get("init") is not None for r in rs) for *_, rs in series):
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for style, color, label, rows in series:
        pts = [(r["gridpoints"], r["init"]) for r in rows if r.get("init") is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                    color=color, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    ax.set_ylabel("Initialization wall time (s)")
    ax.set_title("Initialization overhead vs problem size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


def _routine_tavg(row, timer):
    """tavg (s) of one mpp_clock timer on a row, or None if absent / never hit."""
    t = (row.get("timers") or {}).get(timer)
    return t["tavg"] if (t and t.get("hits", 0) > 0 and t["tavg"] > 0) else None


def plot_routine_isolation(cpu_rows, gpu_rows, timer, outpath, name):
    """Isolate one OpenMP-offloaded routine across problem sizes (two panels).

    * **Left** -- the routine's time as a fraction of the main loop, per
      platform.
    * **Right** -- its CPU-node/GPU speedup (time ratio; > 1.0 = GPU faster),
      matched by job-size index, with the clean 1-node-vs-1-GPU band shaded.

    CAVEAT (on the figure): a GPU routine timer folds in the OpenMP
    `target ... map()` host<->device transfers and runtime overhead, not just
    the kernel; an Nsight Systems run is needed to split compute from transfers.

    Returns None unless at least one branch carries the timer.
    """
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms

    def series(rows):
        return [(r["gridpoints"], _routine_tavg(r, timer) / r["main_loop"],
                 _routine_tavg(r, timer), r["i"])
                for r in rows if _routine_tavg(r, timer)]

    cpu_c, gpu_c = series(cpu_rows), series(gpu_rows)
    if not cpu_c and not gpu_c:
        return None

    boundary = BLOCK * BLOCK * NK * CPU_PER_NODE  # gridpoints at i = 128
    fig, (axf, axs) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: routine as a fraction of the main loop.
    if cpu_c:
        axf.plot([p[0] for p in cpu_c], [100 * p[1] for p in cpu_c],
                 "s-", color="C0", label="CPU node (<=128 ranks)")
    if gpu_c:
        axf.plot([p[0] for p in gpu_c], [100 * p[1] for p in gpu_c],
                 "o-", color="C1", label="1 GPU (A100)")
    axf.set_xscale("log")
    axf.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    axf.set_ylabel(f"{name} time / main loop (%)")
    axf.set_title(f"{name} share of the main loop")
    axf.grid(True, which="both", alpha=0.3)
    axf.legend(loc="best")

    # Right: CPU/GPU speedup (time ratio), matched by job-size index.
    cpu_by_i = {p[3]: p[2] for p in cpu_c}
    pts = sorted((g[0], cpu_by_i[g[3]] / g[2]) for g in gpu_c if g[3] in cpu_by_i)
    if pts:
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        axs.plot(x, y, "o-", color="C2", label=f"{name} CPU/GPU time ratio")
        axs.axhline(1.0, ls="-", color="black", alpha=0.6, lw=1)
        trans = mtransforms.blended_transform_factory(axs.transData, axs.transAxes)
        if max(x) >= boundary:
            axs.axvspan(boundary, max(x), color="C0", alpha=0.08)
            axs.text(boundary, 0.97, " 1 node vs 1 GPU\n (clean comparison)",
                     transform=trans, va="top", ha="left", fontsize=8,
                     color="dimgray")
        for gp, label in REF_POINTS:
            axs.axvline(gp, ls="--", alpha=0.6, color="gray")
            axs.text(gp, 0.02, " " + label, rotation=90, transform=trans,
                     va="bottom", ha="right", fontsize=8, color="gray")
        axs.legend(loc="best")
    else:
        axs.text(0.5, 0.5, "no shared sizes with\nthis timer",
                 transform=axs.transAxes, ha="center", va="center",
                 color="dimgray")
    axs.set_xscale("log")
    axs.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    axs.set_ylabel(f"{name} speedup (CPU node / GPU, >1 = GPU faster)")
    axs.set_title(f"{name} speedup on the GPU")
    axs.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"GPU {name.lower()} timer folds in OpenMP target map() "
                 "host<->device transfers (not bare kernel); "
                 "use Nsight Systems to separate.",
                 fontsize=9, color="dimgray", y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


def clean_name(name):
    """Trim an mpp_clock name to a compact label: '(Ocean foo bar)' -> 'foo bar'.

    Handles names FMS truncated to the column width (no closing paren).
    """
    n = name.strip().strip(" *")
    if n.startswith("("):
        n = n[1:]
    if n.endswith(")"):
        n = n[:-1]
    if n.startswith("Ocean "):
        n = n[len("Ocean "):]
    return n.strip()


# Below this wall-time (s) a routine is at the noise floor; a CPU/GPU ratio
# across it is meaningless (e.g. inter-rank message passing is ~0 on 1 GPU rank).
RATIO_FLOOR_S = 0.05


def _speedup(c, g):
    """CPU/GPU speedup string, or None if either side is below the noise floor."""
    if c is None or g is None or c < RATIO_FLOOR_S or g < RATIO_FLOOR_S:
        return None
    return c / g


def select_components(cpu_row, gpu_row, top_n=12):
    """Top routine-level timers (grain 31) by wall time, CPU and GPU aligned.

    Grain-31 clocks are the main-loop routines (continuity, barotropic stepping,
    viscosity, pressure force, ...). In this configuration they do not nest one
    another -- they sum to `Ocean dynamics` -- so they form a clean per-routine
    breakdown. Returns [(name, cpu_tavg, gpu_tavg), ...] sorted by the larger of
    the two, longest first.
    """
    ct = cpu_row.get("timers", {}) if cpu_row else {}
    gt = gpu_row.get("timers", {}) if gpu_row else {}
    names = set()
    for table in (ct, gt):
        for name, rec in table.items():
            if rec["grain"] == 31 and rec["hits"] > 0 and rec["tavg"] > 0:
                names.add(name)
    comps = [(name, ct.get(name, {}).get("tavg", 0.0),
              gt.get(name, {}).get("tavg", 0.0)) for name in names]
    comps.sort(key=lambda c: max(c[1], c[2]), reverse=True)
    return comps[:top_n]


def plot_breakdown(cpu_row, gpu_row, outpath, top_n=12):
    """Per-routine main-loop time, 1 CPU node vs 1 GPU, at one problem size.

    Paired horizontal bars per routine (longest at top), annotated with the
    CPU/GPU speedup (>1, green = the GPU is faster on that routine; <1, red =
    slower). The whole model is OpenMP-GPU-offloaded, so this is the crux view:
    which offloaded routines map efficiently to one GPU. Continuity does; the
    barotropic solver and viscosity do not, and they dominate -- which is why
    the whole-model main loop regresses.
    """
    import matplotlib.pyplot as plt
    comps = select_components(cpu_row, gpu_row, top_n)
    if not comps:
        return None

    labels = [clean_name(name) for name, _, _ in comps]
    cpu = [c for _, c, _ in comps]
    gpu = [g for _, _, g in comps]
    y = list(range(len(comps)))
    h = 0.38
    xmax = max(max(cpu), max(gpu))

    fig, ax = plt.subplots(figsize=(9, 0.52 * len(comps) + 1.8))
    ax.barh([yi + h / 2 for yi in y], cpu, height=h, color="C0",
            label="CPU node (128 ranks)")
    ax.barh([yi - h / 2 for yi in y], gpu, height=h, color="C1",
            label="1 GPU (A100)")
    for yi, (_, c, g) in zip(y, comps):
        sp = _speedup(c, g)
        if sp is not None:
            ax.text(max(c, g) + xmax * 0.012, yi, f"{sp:.2f}x", va="center",
                    fontsize=8, color=("C2" if sp >= 1 else "C3"))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.15)
    ax.set_xlabel("Main-loop time (s)  --  label = CPU/GPU speedup (>1 GPU faster)")
    ax.set_title("Where the main-loop time goes, by routine")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


# --- report -----------------------------------------------------------------

def fmt_int(n):
    return f"{n:,}" if n is not None else "-"


def fmt_sec(x):
    return f"{x:.3f}" if x is not None else "-"


def cpu_table(rows):
    head = ("| i | ranks | NI x NJ | gridpoints | gp/rank | dt | main loop (s) "
            "| s/step | throughput (cell-up/s) |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        body += (f"| {r['i']} | {r['nranks']} | {r['ni']}x{r['nj']} "
                 f"| {fmt_int(r['gridpoints'])} | {fmt_int(int(r['gridpoints_per_rank']))} "
                 f"| {r['dt']} | {r['main_loop']:.3f} | {r['sec_per_step']:.4f} "
                 f"| {r['throughput']:.3e} |\n")
    return head + body


def gpu_table(rows):
    head = ("| i | NI x NJ | gridpoints | dt | main loop (s) | init (s) | s/step "
            "| throughput (cell-up/s) |\n"
            "|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        body += (f"| {r['i']} | {r['ni']}x{r['nj']} | {fmt_int(r['gridpoints'])} "
                 f"| {r['dt']} | {r['main_loop']:.3f} | {fmt_sec(r['init'])} "
                 f"| {r['sec_per_step']:.4f} | {r['throughput']:.3e} |\n")
    return head + body


def failures_table(failures):
    """Runs that produced no Main loop timer, so they did not complete."""
    if not failures:
        return None
    head = ("| platform | i | NI x NJ | gridpoints | log | cause (from stderr) |\n"
            "|---|---|---|---|---|---|\n")
    body = ""
    for f in failures:
        reason = f.get("reason") or "_(no stderr captured)_"
        body += (f"| {f['platform']} | {f['i']} | {f['ni']}x{f['nj']} "
                 f"| {fmt_int(f['gridpoints'])} | `{f['fname']}` | {reason} |\n")
    return head + body


def breakdown_table(cpu_row, gpu_row, top_n=12):
    """Per-routine CPU-node vs GPU times at one problem size, longest first."""
    comps = select_components(cpu_row, gpu_row, top_n)
    if not comps:
        return None
    head = ("| routine | CPU node (s) | 1 A100 (s) | speedup (CPU/GPU) |\n"
            "|---|---|---|---|\n")
    body = ""
    for name, c, g in comps:
        sp = _speedup(c, g)
        sp_str = f"{sp:.2f}x" if sp is not None else "n/a"
        body += (f"| {clean_name(name)} | {c:.3f} | {g:.3f} | {sp_str} |\n")
    return head + body


def prefix_table(cpu_row, gpu_row, prefix, top_n=8):
    """Per-sub-timer CPU-node vs GPU table for all timers sharing a name prefix.

    Used to open up the barotropic solver into its sub-steps (BT pre-calcs, halo
    updates, ...) so the bottleneck is visible. Longest (by max side) first.
    """
    ct = (cpu_row.get("timers") or {})
    gt = (gpu_row.get("timers") or {})
    names = {n for n in set(ct) | set(gt) if n.startswith(prefix)}
    rows = []
    for n in names:
        c = ct.get(n, {}).get("tavg", 0.0)
        g = gt.get(n, {}).get("tavg", 0.0)
        if (ct.get(n, {}).get("hits", 0) or gt.get(n, {}).get("hits", 0)):
            rows.append((n, c, g))
    rows.sort(key=lambda r: max(r[1], r[2]), reverse=True)
    rows = rows[:top_n]
    if not rows:
        return None
    head = ("| sub-step | CPU node (s) | 1 A100 (s) | speedup (CPU/GPU) |\n"
            "|---|---|---|---|\n")
    body = ""
    for n, c, g in rows:
        sp = _speedup(c, g)
        sp_str = f"{sp:.2f}x" if sp is not None else "n/a"
        body += f"| {clean_name(n)} | {c:.3f} | {g:.3f} | {sp_str} |\n"
    return head + body


def comparison_table(cpu_rows, gpu_rows):
    """Head-to-head over the job sizes present in BOTH branches.

    One row per shared `i`: the same problem size run on 1 full CPU node vs 1
    GPU. `GPU/CPU speedup` is GPU throughput / CPU throughput -- > 1 means the
    GPU wins at that size. Returns None if the branches share no job sizes.
    """
    cpu_by_i = {r["i"]: r for r in cpu_rows}
    shared = [(cpu_by_i[g["i"]], g) for g in gpu_rows if g["i"] in cpu_by_i]
    if not shared:
        return None
    head = ("| i | gridpoints | CPU ranks | CPU loop (s) | GPU loop (s) "
            "| CPU thrpt (cell-up/s) | GPU thrpt (cell-up/s) | GPU/CPU speedup |\n"
            "|---|---|---|---|---|---|---|---|\n")
    body = ""
    for c, g in shared:
        speedup = g["throughput"] / c["throughput"]
        body += (f"| {g['i']} | {fmt_int(g['gridpoints'])} | {c['nranks']} "
                 f"| {c['main_loop']:.3f} | {g['main_loop']:.3f} "
                 f"| {c['throughput']:.3e} | {g['throughput']:.3e} "
                 f"| {speedup:.2f}x |\n")
    return head + body


def continuity_table(cpu_rows, gpu_rows):
    """Head-to-head on the continuity solver alone.

    One row per job size carrying a continuity timer on both branches. The
    `%loop` columns show how much of each branch's main loop the solver is;
    `GPU/CPU` is the continuity-only throughput ratio.
    Returns None if no shared size has continuity timings on both branches.
    """
    cpu_by_i = {r["i"]: r for r in cpu_rows if r.get("continuity")}
    shared = [(cpu_by_i[g["i"]], g) for g in gpu_rows
              if g.get("continuity") and g["i"] in cpu_by_i]
    if not shared:
        return None
    head = ("| i | gridpoints | CPU cont (s) | GPU cont (s) | CPU %loop "
            "| GPU %loop | GPU/CPU continuity speedup |\n"
            "|---|---|---|---|---|---|---|\n")
    body = ""
    for c, g in shared:
        speedup = g["continuity_throughput"] / c["continuity_throughput"]
        body += (f"| {g['i']} | {fmt_int(g['gridpoints'])} "
                 f"| {c['continuity']:.3f} | {g['continuity']:.3f} "
                 f"| {100 * c['continuity_frac']:.1f}% "
                 f"| {100 * g['continuity_frac']:.1f}% "
                 f"| {speedup:.2f}x |\n")
    return head + body


def write_csv(rows, path):
    if not rows:
        return
    cols = ["platform", "i", "ni", "nj", "nk", "nranks", "gridpoints",
            "gridpoints_per_rank", "dt", "nsteps", "main_loop", "total_runtime",
            "init", "termination", "sec_per_step", "throughput",
            "continuity", "continuity_per_step", "continuity_throughput",
            "continuity_frac"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_report(cpu_rows, gpu_rows, plots, nsteps, title, prov=None,
                 failures=None, breakdown=None):
    # per-rank work for the weak-scaling regime: one 32x32 block x NK levels
    block_str = f"{BLOCK}x{BLOCK}x{NK}"
    L = []
    L.append(f"# {title}\n")

    if prov is not None:
        L.append(render_stamp(prov))

    L.append("## Intent\n")
    L.append(
        "Characterize MOM6 throughput on Derecho for the `double_gyre` "
        "benchmark, comparing a single A100 GPU against a full 128-rank CPU "
        "node. The GPU build offloads the model **whole** via OpenMP "
        "target directives (`-mp=gpu`): the dynamical core, tracers, and "
        "parameterizations are annotated with `!$omp target teams loop` regions "
        "and explicit `map()` host<->device data movement (~28 source files).")

    # Headline reads straight off the per-routine breakdown, when we have it.
    if breakdown:
        bi, bc, bg = breakdown
        cont_c = bc.get("timers", {}).get(CONTINUITY_TIMER, {}).get("tavg")
        cont_g = bg.get("timers", {}).get(CONTINUITY_TIMER, {}).get("tavg")
        if cont_c and cont_g:
            L.append("## Key finding\n")
            L.append(
                f"At the clean 1-node-vs-1-GPU comparison (i={bi}, "
                f"{fmt_int(bc['gridpoints'])} gridpoints): the continuity solver "
                f"is **{cont_c / cont_g:.2f}x faster** on one A100 than on a full "
                f"CPU node ({cont_c:.1f}s -> {cont_g:.1f}s, transfers included), "
                f"yet the **whole-model main loop is "
                f"{bg['main_loop'] / bc['main_loop']:.2f}x slower** "
                f"({bc['main_loop']:.1f}s -> {bg['main_loop']:.1f}s). The whole "
                "model is OpenMP-offloaded, so this is not an Amdahl/un-ported "
                "story: some offloaded routines map well to the GPU (continuity, "
                "pressure force, momentum increments), but others -- the "
                "**barotropic solver above all**, plus viscosity -- are much "
                "slower on the GPU and dominate the loop. See the breakdown "
                "below.\n")

    L.append("## Methodology\n")
    L.append(
        f"Each run advances exactly **{nsteps} dynamic steps** "
        "(`TIMEUNIT = dt` with `DAYMAX = 150`), so wall-clock time is directly "
        "comparable across problem sizes. The job-size index `i` sets a "
        "near-square layout of `i` 32x32 column blocks at NK=100.\n\n"
        "- **CPU branch** (`run-scaling-sweep.sh cpu`): weak scaling. Ranks grow "
        f"with `i` at a constant {block_str} gridpoints/rank up to the 128-rank "
        "node cap; beyond that, ranks stay at 128 and per-rank work grows.\n"
        "- **GPU branch** (`run-scaling-sweep.sh gpu`): single-device problem-size scan "
        "(1 GPU, 1x1 decomposition) that reveals the throughput-saturation "
        "knee.\n\n"
        "The per-run measurement is the cross-PE mean of the FMS `Main loop` "
        "timer. Throughput is reported as cell-updates/s "
        f"= gridpoints x {nsteps} / main-loop-time. With `clock_grain = 'ROUTINE'` "
        "in `input.nml` the runs also emit per-routine timers, which drive the "
        "continuity-isolation and where-the-time-goes sections below.\n")

    if plots.get("cpu_timing"):
        L.append("## CPU timing\n")
        L.append(f"![CPU timing]({os.path.basename(plots['cpu_timing'])})\n")
        L.append(
            "One continuous problem-size scan. **Left of the divider** "
            f"(weak scaling): ranks grow with the problem at a fixed {block_str} "
            "gridpoints/rank, so the rise is halo-exchange plus "
            "barotropic-solver collective overhead. **Right of the divider** "
            "(saturated node): ranks stay at the 128-rank cap and per-rank "
            "work grows, so time rises with the added work. The slope change "
            "at the divider is the transition between the two regimes.\n")

    if plots.get("throughput"):
        L.append("## Throughput vs problem size\n")
        L.append(f"![Throughput]({os.path.basename(plots['throughput'])})\n")
        L.append(
            "Dashed verticals mark the production operating points "
            "(19.4M gridpoints/GPU without MARBL; 9.7M with MARBL).\n")

    if plots.get("speedup"):
        L.append("## GPU vs CPU speedup\n")
        L.append(f"![Speedup]({os.path.basename(plots['speedup'])})\n")
        L.append(
            "Speedup is the single-GPU throughput divided by the CPU-node "
            "throughput at the same problem size (matched by job-size index "
            "`i`). The solid line at 1.0 is parity: above it the GPU wins, "
            "below it a full CPU node wins. The shaded band (problem size at or "
            "beyond the i=128 node-saturation point) is where the comparison is "
            "a clean 1-GPU-vs-1-full-node match; left of it the CPU branch is "
            "still weak-scaling across fewer than 128 ranks, so the ratio there "
            "mixes scaling regimes and should be read with care. Dashed "
            "verticals mark the production operating points.\n")

    ct = comparison_table(cpu_rows, gpu_rows)
    if ct:
        L.append("## Head-to-head: 1 GPU vs 1 CPU node\n")
        L.append(
            "Job sizes present in both branches, putting one full CPU node and "
            "one A100 on the identical problem. `GPU/CPU speedup` > 1 means the "
            "GPU wins at that size.\n")
        L.append(ct)

    if breakdown and (plots.get("breakdown") or breakdown_table(*breakdown[1:])):
        bi, bc, bg = breakdown
        L.append("## Where the main-loop time goes (by routine)\n")
        L.append(
            f"Per-routine FMS timers (grain 31) at the clean comparison point "
            f"i={bi} ({fmt_int(bc['gridpoints'])} gridpoints, "
            f"{bc['ni']}x{bc['nj']}), 1 full CPU node vs 1 A100. `speedup` is "
            "CPU/GPU time, so > 1 means the GPU is faster on that routine.\n")
        if plots.get("breakdown"):
            L.append(f"![Breakdown]({os.path.basename(plots['breakdown'])})\n")
        bt = breakdown_table(bc, bg)
        if bt:
            L.append(bt)
        L.append(
            "\nEvery routine here is OpenMP-GPU-offloaded; the spread is in how "
            "well each maps to one GPU. **Continuity is the clearest win** (and on "
            "the CPU it is the single largest cost). The whole-model regression is "
            "driven by routines that offload *poorly* -- the **barotropic solver "
            "most of all** (slowest on the GPU despite having the most `target` "
            "regions in the source), plus horizontal viscosity and Coriolis. The "
            "barotropic mode is an iterative sub-cycle fired many times per "
            "baroclinic step, so it is launch/overhead-bound on the GPU (see the "
            "barotropic section below for which sub-step is responsible). Note "
            "also the diagnostics/`Ocean Other` block (not a grain-31 routine) "
            "balloons on the GPU. `message passing` is inter-rank halo exchange -- "
            "the lone GPU rank has none, so its near-zero time is structural "
            "(ratio `n/a`), not a GPU win. Every GPU routine time still includes "
            "its `target map()` host<->device transfers (see the caveat below).\n")

    bt_sub = prefix_table(breakdown[1], breakdown[2], BAROTROPIC_PREFIX) if breakdown else None
    if plots.get("barotropic") or bt_sub:
        L.append("## Barotropic solver -- the main GPU bottleneck\n")
        L.append(
            "The barotropic solver is the largest single cost on the GPU and the "
            "biggest drag on the whole-model number, so it is the prime "
            "optimization target. It is an explicit free-surface sub-cycle: many "
            "short barotropic steps per baroclinic step, each an offloaded "
            "`!$omp target` region. That structure -- a long sequence of small "
            "kernels -- is what maps poorly to the GPU.\n")
        if plots.get("barotropic"):
            L.append(f"![Barotropic]({os.path.basename(plots['barotropic'])})\n")
        if bt_sub:
            bi = breakdown[0]
            L.append(
                f"Sub-steps at i={bi} ({fmt_int(breakdown[1]['gridpoints'])} "
                "gridpoints), 1 CPU node vs 1 A100:\n")
            L.append(bt_sub)
            L.append(
                "\nThe cost is concentrated in **`BT pre-calcs`** (the per-substep "
                "barotropic calculation), which is several times slower on the "
                "GPU. Crucially, the **halo updates are ~0 s on the GPU** (one "
                "rank, no inter-rank exchange), so the bottleneck is **not** "
                "communication -- it is the per-substep compute kernels firing "
                "with too little work each (launch/overhead-bound). The deficit "
                "eases at larger problem sizes (more work per kernel), pointing to "
                "kernel-fusion / fewer-larger-launches across the sub-cycle as the "
                "optimization, and Nsight Systems on the barotropic region as the "
                "next measurement.\n")

    cont_t = continuity_table(cpu_rows, gpu_rows)
    if plots.get("continuity") or cont_t:
        L.append("## Continuity solver in isolation\n")
        L.append(
            "The continuity solver is the reviewer's focus, the single largest "
            "routine in the CPU main loop (~35-50%), and -- per the breakdown "
            "above -- one of the routines that offloads *well*. This section "
            "isolates it across problem sizes using MOM6's "
            "`(Ocean continuity equation)` timer, exposed by setting "
            "`clock_grain = 'ROUTINE'` in `input.nml`.\n\n"
            "> **Caveat -- what this timer measures.** The continuity timer is an "
            "FMS `mpp_clock` around the solver *call*. On the GPU it folds in the "
            "OpenMP **`target ... map()` host<->device transfers** and runtime "
            "overhead around the offloaded loops, not just the kernel. It is "
            "therefore the right figure for *what the model actually pays for this "
            "routine end-to-end*, but it **overstates the kernel** and should not "
            "be read as GPU compute time. Splitting kernel time from the "
            "host<->device transfers needs an Nsight Systems run "
            "(`run-profile.sh`).\n")
        if plots.get("continuity"):
            L.append(f"![Continuity]({os.path.basename(plots['continuity'])})\n")
            L.append(
                "**Left:** continuity as a fraction of the main loop. On the "
                "**CPU** it is ~35-50% (the single largest routine). On the "
                "**GPU** the same solver is a much smaller share -- not because it "
                "is slower but because the other offloaded routines (barotropic, "
                "viscosity) inflate the GPU's denominator. **Right:** "
                "continuity-only throughput ratio (GPU / CPU node) at matched "
                "sizes; the GPU wins this routine by ~2x in the clean-comparison "
                "band (more at small sizes, where the CPU side is only a few "
                "ranks), transfers included.\n")
        if cont_t:
            L.append(cont_t)

    if plots.get("init"):
        L.append("## Initialization overhead\n")
        L.append(f"![Initialization]({os.path.basename(plots['init'])})\n")
        L.append(
            "The FMS `Initialization` timer -- setup, allocation, and (on the "
            "GPU) host-to-device staging and kernel setup before the main loop. "
            "This is a fixed per-run cost that the main-loop throughput numbers "
            "do not capture. The CPU node stays modest while the GPU's init "
            "cost climbs steeply with problem size, which both drags the GPU at "
            "small problems and tracks its march toward the single-device "
            "memory ceiling at large ones.\n")

    ft = failures_table(failures)
    if ft:
        L.append("## Failed / missing runs\n")
        L.append(
            "These runs produced no FMS `Main loop` timer, so they did not "
            "complete and are excluded from the plots and tables above. The "
            "`cause` column is the failing line from the run's stderr. A GPU "
            "failure only at the largest size is the single-device memory "
            "ceiling: one A100 (40 GB) cannot hold the problem, and the "
            "allocation that tips it over aborts on the first dynamic step.\n")
        L.append(ft)

    L.append("## Results: CPU branch\n")
    L.append(cpu_table(cpu_rows) if cpu_rows else "_No CPU runs found._\n")

    L.append("\n## Results: GPU branch\n")
    if gpu_rows:
        L.append(gpu_table(gpu_rows))
    else:
        L.append("_GPU runs pending. Re-run this script with `--gpu-dir` once "
                 "the queue job completes to fill in this section._\n")

    if prov is not None:
        L.append(render_provenance(prov, include_stamp=False))

    return "\n".join(L) + "\n"


DEFAULT_REPORTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "reports"))


def resolve_outdir(args, now):
    """Pick the report directory.

    With --outdir, honor it verbatim (explicit escape hatch). Otherwise build
    `<reports-dir>/<YYYY-MM-DD-HHMMSS>-<label>` -- the seconds-resolution stamp
    keeps multiple reports per day distinct and chronologically sorted. On the
    rare same-second collision, append -2, -3, ... so we never clobber.
    """
    if args.outdir:
        return args.outdir
    stamp = now.strftime("%Y-%m-%d-%H%M%S")
    base = os.path.join(args.reports_dir, f"{stamp}-{args.label}")
    outdir, n = base, 1
    while os.path.exists(outdir):
        n += 1
        outdir = f"{base}-{n}"
    return outdir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-dir", help="directory with cpu_*.out logs")
    ap.add_argument("--gpu-dir", help="directory with gpu_*.out logs")
    ap.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR,
                    help="parent directory for timestamped report dirs "
                    "(default: ../reports)")
    ap.add_argument("--label", default="double_gyre",
                    help="trailing label for the report dir name "
                    "(default: double_gyre); dir is <date-time>-<label>")
    ap.add_argument("--outdir", help="explicit output directory; overrides the "
                    "timestamped --reports-dir/--label naming")
    ap.add_argument("--nsteps", type=int, default=NSTEPS_DEFAULT,
                    help=f"dynamic steps per run (default {NSTEPS_DEFAULT})")
    ap.add_argument("--breakdown-i", type=int, default=BREAKDOWN_I_DEFAULT,
                    help="job-size index for the per-routine breakdown "
                    f"(default {BREAKDOWN_I_DEFAULT}); falls back to the largest "
                    "shared size if absent")
    ap.add_argument("--title", default="MOM6 double_gyre GPU vs CPU scaling")
    ap.add_argument("--stack-dir", help="path to turbo-stack checkout, for "
                    "recording commit hashes and build flags in the report")
    ap.add_argument("--note", help="free-form note added to the provenance block")
    ap.add_argument("--date", help="override the generated timestamp text in "
                    "the provenance block; defaults to now (YYYY-MM-DD HH:MM:SS)")
    args = ap.parse_args()

    if not args.cpu_dir and not args.gpu_dir:
        ap.error("provide at least one of --cpu-dir / --gpu-dir")

    now = datetime.datetime.now()
    gen_time = args.date or now.isoformat(sep=" ", timespec="seconds")

    cpu_rows, cpu_fail = collect(args.cpu_dir, "cpu")
    gpu_rows, gpu_fail = collect(args.gpu_dir, "gpu")
    add_throughput(cpu_rows, args.nsteps)
    add_throughput(gpu_rows, args.nsteps)
    failures = cpu_fail + gpu_fail
    print(f"Parsed {len(cpu_rows)} CPU run(s), {len(gpu_rows)} GPU run(s); "
          f"{len(failures)} failed/incomplete.")

    prov = gather_provenance(args.stack_dir, args.note, gen_time)

    outdir = resolve_outdir(args, now)
    os.makedirs(outdir, exist_ok=True)
    write_csv(cpu_rows + gpu_rows, os.path.join(outdir, "results.csv"))
    with open(os.path.join(outdir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    plots = {}
    if cpu_rows:
        plots["cpu_timing"] = plot_cpu_timing(
            cpu_rows, os.path.join(outdir, "cpu_timing.png"))
    if cpu_rows or gpu_rows:
        plots["throughput"] = plot_throughput(
            cpu_rows, gpu_rows, os.path.join(outdir, "throughput.png"))
    if cpu_rows and gpu_rows:
        plots["speedup"] = plot_speedup(
            cpu_rows, gpu_rows, os.path.join(outdir, "speedup.png"))
    if cpu_rows or gpu_rows:
        plots["init"] = plot_init(
            cpu_rows, gpu_rows, os.path.join(outdir, "init.png"))
    if cpu_rows or gpu_rows:
        plots["continuity"] = plot_routine_isolation(
            cpu_rows, gpu_rows, CONTINUITY_TIMER,
            os.path.join(outdir, "continuity.png"), "Continuity")
        plots["barotropic"] = plot_routine_isolation(
            cpu_rows, gpu_rows, BAROTROPIC_TIMER,
            os.path.join(outdir, "barotropic.png"), "Barotropic solver")

    # Per-routine breakdown at one shared problem size (timers required on both).
    breakdown = None
    cpu_by_i = {r["i"]: r for r in cpu_rows if r.get("timers")}
    gpu_by_i = {r["i"]: r for r in gpu_rows if r.get("timers")}
    shared = sorted(set(cpu_by_i) & set(gpu_by_i))
    if shared:
        bi = args.breakdown_i if args.breakdown_i in shared else max(shared)
        breakdown = (bi, cpu_by_i[bi], gpu_by_i[bi])
        plots["breakdown"] = plot_breakdown(
            cpu_by_i[bi], gpu_by_i[bi], os.path.join(outdir, "breakdown.png"))

    report = build_report(cpu_rows, gpu_rows, plots, args.nsteps, args.title,
                          prov=prov, failures=failures, breakdown=breakdown)
    report_path = os.path.join(outdir, "report.md")
    with open(report_path, "w") as fh:
        fh.write(report)
    print(f"Wrote report to {outdir}/")
    print(f"  report.md, results.csv, provenance.json")
    for p in plots.values():
        if p:
            print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
