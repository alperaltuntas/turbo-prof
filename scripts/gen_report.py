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


def plot_continuity(cpu_rows, gpu_rows, outpath):
    """Continuity-solver-only view: the GPU-ported region, not the whole model.

    Two panels share the problem-size x-axis:

    * **Left** -- continuity time as a fraction of the main loop, for each
      platform. This is the Amdahl ceiling: only this slice of the model is
      GPU-ported, so the whole-model speedup can never exceed what this slice
      allows. It also shows the port shifting the balance between the branches.
    * **Right** -- continuity-only speedup (GPU continuity throughput / CPU
      continuity throughput), matched by job-size index. A solid line at 1.0 is
      parity. This isolates the port's own performance from the un-ported model.

    CAVEAT (stated on the figure): the GPU continuity timer is an FMS clock
    around the call -- it captures the AMReX call stack plus device<->host
    copies, not the bare kernel. It is the right number for "what the model
    pays for the ported region", but it overstates the kernel; an Nsight
    Systems run is needed to split compute from transfers.

    Returns None unless at least one branch carries continuity timings.
    """
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms

    cpu_c = [r for r in cpu_rows if r.get("continuity")]
    gpu_c = [r for r in gpu_rows if r.get("continuity")]
    if not cpu_c and not gpu_c:
        return None

    boundary = BLOCK * BLOCK * NK * CPU_PER_NODE  # gridpoints at i = 128
    fig, (axf, axs) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: continuity as a fraction of the main loop.
    if cpu_c:
        axf.plot([r["gridpoints"] for r in cpu_c],
                 [100 * r["continuity_frac"] for r in cpu_c],
                 "s-", color="C0", label="CPU node (<=128 ranks)")
    if gpu_c:
        axf.plot([r["gridpoints"] for r in gpu_c],
                 [100 * r["continuity_frac"] for r in gpu_c],
                 "o-", color="C1", label="1 GPU (A100)")
    axf.set_xscale("log")
    axf.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    axf.set_ylabel("Continuity time / main loop (%)")
    axf.set_title("Ported share of the main loop (Amdahl ceiling)")
    axf.grid(True, which="both", alpha=0.3)
    axf.legend(loc="best")

    # Right: continuity-only GPU/CPU speedup, matched by job-size index.
    cpu_by_i = {r["i"]: r for r in cpu_c}
    pts = sorted((g["gridpoints"],
                  g["continuity_throughput"] / cpu_by_i[g["i"]]["continuity_throughput"])
                 for g in gpu_c if g["i"] in cpu_by_i)
    if pts:
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        axs.plot(x, y, "o-", color="C2", label="GPU / CPU continuity throughput")
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
        axs.legend(loc="upper left")
    else:
        axs.text(0.5, 0.5, "no shared sizes with\ncontinuity timings",
                 transform=axs.transAxes, ha="center", va="center",
                 color="dimgray")
    axs.set_xscale("log")
    axs.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    axs.set_ylabel("Continuity-only speedup (GPU / CPU node)")
    axs.set_title("Continuity-solver speedup (port's own region)")
    axs.grid(True, which="both", alpha=0.3)

    fig.suptitle("GPU continuity timer = AMReX call stack + device<->host "
                 "copies (not bare kernel); use Nsight Systems to separate.",
                 fontsize=9, color="dimgray", y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
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
    head = ("| platform | i | NI x NJ | gridpoints | log |\n"
            "|---|---|---|---|---|\n")
    body = ""
    for f in failures:
        body += (f"| {f['platform']} | {f['i']} | {f['ni']}x{f['nj']} "
                 f"| {fmt_int(f['gridpoints'])} | `{f['fname']}` |\n")
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
    """Head-to-head on the GPU-ported continuity solver alone.

    One row per job size carrying a continuity timer on both branches. The
    `%loop` columns show how much of each branch's main loop the solver is
    (the Amdahl ceiling); `GPU/CPU` is the continuity-only throughput ratio.
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
                 failures=None):
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
        "node, to locate the problem size at which the GPU port becomes "
        "competitive and to confirm where production-scale workloads "
        "(~19.4M gridpoints/GPU without MARBL, ~9.7M with MARBL) land on the "
        "curve.\n")

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
        f"= gridpoints x {nsteps} / main-loop-time.\n")

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

    cont_t = continuity_table(cpu_rows, gpu_rows)
    if plots.get("continuity") or cont_t:
        L.append("## Continuity solver in isolation (the ported region)\n")
        L.append(
            "The whole-model numbers above are governed by Amdahl's law: only "
            "the **continuity solver** is GPU-ported, so most of the main loop "
            "(tracers, thermodynamics, the barotropic solver, diagnostics, halo "
            "updates) still runs on the host. This section isolates the ported "
            "region using MOM6's `(Ocean continuity equation)` timer, exposed by "
            "setting `clock_grain = 'ROUTINE'` in `input.nml`.\n\n"
            "> **Caveat -- what this timer measures.** The continuity timer is an "
            "FMS `mpp_clock` around the solver *call*. On the GPU it captures the "
            "**AMReX call stack plus the device<->host copies**, not the bare "
            "kernel. In the current architecture the rest of the model is on the "
            "host, so data ping-pongs across the PCIe bus every step and that "
            "transfer cost is folded into this number. It is therefore the right "
            "figure for *what the model actually pays for the ported region "
            "end-to-end*, but it **overstates the kernel** and should not be read "
            "as GPU compute time. Separating kernel / copy / AMReX overhead needs "
            "an Nsight Systems run (`run-profile.sh`).\n")
        if plots.get("continuity"):
            L.append(f"![Continuity]({os.path.basename(plots['continuity'])})\n")
            L.append(
                "**Left:** continuity as a fraction of the main loop -- the "
                "Amdahl ceiling on whole-model speedup. **Right:** continuity-only "
                "throughput ratio (GPU / CPU node) at matched problem sizes; above "
                "the 1.0 line the GPU wins on its own region even with copies "
                "included.\n")
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
            "These runs produced no FMS `Main loop` timer -- the log never "
            "reached the timer table, so the run did not complete and is "
            "excluded from the plots and tables in this report. A failure only "
            "at the largest GPU size is the expected single-device memory "
            "ceiling (the GPU ran out of memory); confirm against the run log.\n")
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
        plots["continuity"] = plot_continuity(
            cpu_rows, gpu_rows, os.path.join(outdir, "continuity.png"))

    report = build_report(cpu_rows, gpu_rows, plots, args.nsteps, args.title,
                          prov=prov, failures=failures)
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
