#!/usr/bin/env python3
"""Generate a Markdown report from a MOM6 double_gyre four-config comparison sweep.

The companion sweep script (run-compare-sweep.sh, usually via job-compare-cpu.sh
and job-compare-gpu.sh) runs four executable configurations --
{dev_turbo, iturbo_amrex} x {CPU, GPU} -- across the standard
job-size sweep, repeating each (config, size) point N times. It leaves
`<config>_<i0>_run<r>.out` logs (plus `.err` / `.stats`) in the run directory.

Per run we read the FMS mpp_clock table (cross-PE "tavg") and, for AMReX-mode
runs, the AMReX TinyProfiler INCLUSIVE table; repeats are averaged. The five
ported continuity PPM kernels are timed by MOM6 cpu_clock in dev/turbo builds
and by TinyProfiler in iturbo AMReX builds (each falls back to the other), so
they are comparable wall-clock figures.

This is the "comparison sweep" report type. Reusable building blocks (log
parsing, provenance capture) live in the `turboprof` package; only the
comparison-specific plots and tables live here.

The report is **facts only** -- run parameters, plots, and data tables. The
interpretation (conclusions, bottleneck calls, next steps) is added afterwards
by the report-commentary skill, which fills the `<!-- commentary: NAME -->`
anchors this script leaves in the Markdown. See `docs/REPORTING.md`.

Usage (under an env with matplotlib, e.g. conda `npl`):

    python3 gen_compare_report.py --run-dir DIR [--stack NAME=PATH ...] \
        [--outdir DIR] [--nsteps 150] [--title "..."]
"""

import argparse
import csv
import datetime
import hashlib
import json
import os

from turboprof.parsing import (
    NK, BLOCK, CPU_PER_NODE, NSTEPS_DEFAULT, KERNELS,
    collect_compare, add_throughput)
from turboprof.provenance import (
    gather_provenance_multi, render_provenance_multi, render_stamp)
from turboprof.reporting import DEFAULT_REPORTS_DIR, resolve_outdir

# The four configurations, in display order (CPU group then GPU group). For each:
#   tag         : log-file prefix used by run-compare-sweep.sh
#   abbrev      : short column header (dt=dev/turbo, it=iturbo; C=CPU, G=GPU;
#                 ax=AMReX kernels)
#   platform    : cpu (weak-scaling ranks) or gpu (1 rank, 1 device)
#   prefer_tiny : primary per-kernel profiler -- TinyProfiler for iturbo,
#                 MOM6 cpu_clock for dev/turbo (the other is the fallback)
#   stack       : which --stack entry built this config's executable
CONFIGS = [
    ("dev_turbo_CPU",      "dt_C",    "cpu", False, "dev-turbo-cpu"),
    ("iturbo_CPU_amrex",   "it_C_ax", "cpu", True,  "iturbo-cpu"),
    ("dev_turbo_GPU",      "dt_G",    "gpu", False, "dev-turbo"),
    ("iturbo_GPU_amrex",   "it_G_ax", "gpu", True,  "iturbo"),
]
ABBREV = {c[0]: c[1] for c in CONFIGS}

# Human-facing labels used in every plot legend (and surfaced in the config
# table so the mapping from each config's branch/build to its legend label is
# explicit). Encodes the kernel language and -- for the dev/turbo GPU build --
# how the Fortran is offloaded.
DISPLAY = {
    "dev_turbo_CPU":    "Fortran (CPU)",
    "iturbo_CPU_amrex": "AMReX/C++ (CPU)",
    "dev_turbo_GPU":    "Fortran [do concurrent + OMP target] (GPU)",
    "iturbo_GPU_amrex": "AMReX/C++ (GPU)",
}

DEFAULT_STACKS = {
    "dev-turbo-cpu": "/glade/work/altuntas/turbo-stack-dev-turbo-cpu",
    "iturbo-cpu":    "/glade/work/altuntas/turbo-stack-iturbo-cpu",
    "dev-turbo":     "/glade/work/altuntas/turbo-stack-dev-turbo",
    "iturbo":        "/glade/work/altuntas/turbo-stack-iturbo",
}

# Speedups to report (label, baseline tag, config tag): each iturbo variant
# relative to its dev/turbo baseline on the same hardware. >1 = faster.
SPEEDUPS = [
    ("it_C_ax", "dev_turbo_CPU", "iturbo_CPU_amrex"),
    ("it_G_ax", "dev_turbo_GPU", "iturbo_GPU_amrex"),
]

# Per-config plot styling: color encodes platform (CPU=blue vs GPU=orange),
# marker + linestyle encode the build variant (dev/turbo = solid square vs
# iturbo = dashed circle). Kept consistent across every plot.
_STYLE = {
    "dev_turbo_CPU":      dict(color="C0", marker="s", ls="-"),
    "iturbo_CPU_amrex":   dict(color="C0", marker="o", ls="--"),
    "dev_turbo_GPU":      dict(color="C1", marker="s", ls="-"),
    "iturbo_GPU_amrex":   dict(color="C1", marker="o", ls="--"),
}

# Reference markers shared by the size-axis plots, mirroring gen_report.py:
#  - REF_POINTS: production operating points (gridpoints, label).
#  - the i=128 size (BLOCK^2 * NK * CPU_PER_NODE = 13.1M gridpoints) where the
#    CPU run saturates a full Derecho CPU node (128 cores), giving a clean
#    "128 CPU cores vs 1 GPU" comparison. (Note: a Derecho GPU node has only 64
#    cores + 4 A100s, so this is a core-count comparison, not node-vs-node.)
REF_POINTS = [
    (540 * 480 * 75, "540x480x75 = 19.4M (production)"),
]


def _mark_reference_points(ax, node_line=True, text=True):
    """Overlay the production operating-point verticals and, optionally, the
    1-node-vs-1-GPU clean-comparison boundary. Call after plotting the data."""
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    if node_line:
        boundary = BLOCK * BLOCK * NK * CPU_PER_NODE  # gridpoints at i = 128
        ax.axvline(boundary, ls=":", color="red", alpha=0.7)
        if text:
            ax.text(boundary, 0.02, " 128 CPU cores vs 1 GPU",
                    rotation=90, transform=trans, va="bottom", ha="right",
                    fontsize=8, color="indianred")
    for gp, label in REF_POINTS:
        ax.axvline(gp, ls="--", alpha=0.6, color="gray")
        if text:
            ax.text(gp, 0.02, " " + label, rotation=90, transform=trans,
                    va="bottom", ha="right", fontsize=8, color="gray")


def _annotate_scaling(ax):
    """Mark the CPU scaling regimes straddling the i=128 boundary: left of it
    ranks grow with the problem (weak scaling), right of it ranks are pinned at
    128 and per-rank work grows (saturated node). Applies to the CPU configs
    only -- the GPU configs run on 1 device throughout."""
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    boundary = BLOCK * BLOCK * NK * CPU_PER_NODE  # gridpoints at i = 128
    ax.text(boundary, 0.93, "<- CPU weak scaling  ", transform=trans,
            ha="right", va="top", fontsize=8, color="indianred", style="italic")
    ax.text(boundary, 0.93, "  CPU 128 ranks fixed ->", transform=trans,
            ha="left", va="top", fontsize=8, color="indianred", style="italic")
    ax.text(0.075, 0.015, "GPU: 1 device throughout", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color="dimgray", style="italic")


# --- plots ------------------------------------------------------------------

def plot_throughput(rows_by_cfg, outpath):
    """Throughput vs problem size, one line per configuration (log-log)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    drawn = False
    for tag, _, _, _, _ in CONFIGS:
        rows = rows_by_cfg.get(tag)
        if not rows:
            continue
        drawn = True
        ax.plot([r["gridpoints"] for r in rows], [r["throughput"] for r in rows],
                label=DISPLAY[tag], **_STYLE[tag])
    if not drawn:
        plt.close(fig)
        return None
    _mark_reference_points(ax)
    _annotate_scaling(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    ax.set_ylabel("Throughput (cell-updates / s)")
    ax.set_title("Whole-model throughput vs problem size "
                 "(iturbo includes bridge)")
    # Headroom so the weak-scaling annotation near the top doesn't overlay data.
    ax.set_ylim(top=ax.get_ylim()[1] * 3.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


def plot_kernel_throughput(rows_by_cfg, outpath, nsteps):
    """Compute-only throughput: cell-updates/s derived from the continuity
    kernel COMPUTE time (outer zonal+meridional edge_thickness), excluding the
    AMReX bridge marshalling. The kernels-only analog of plot_throughput; for
    iturbo the gap between the two is the bridge overhead."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    drawn = False
    for tag, _, _, _, _ in CONFIGS:
        pts = []
        for r in rows_by_cfg.get(tag, []):
            kt = _kernel_only_time(r)
            if kt:
                pts.append((r["gridpoints"], r["gridpoints"] * nsteps / kt))
        pts.sort()
        if not pts:
            continue
        drawn = True
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                label=DISPLAY[tag], **_STYLE[tag])
    if not drawn:
        plt.close(fig)
        return None
    _mark_reference_points(ax)
    _annotate_scaling(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    ax.set_ylabel("Kernel-only throughput (cell-updates / s)")
    ax.set_title("Continuity kernel throughput (excludes bridge)")
    ax.set_ylim(top=ax.get_ylim()[1] * 4.5)  # headroom for the scaling labels
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


def plot_continuity(rows_by_cfg, outpath):
    """Continuity-solver wall time vs problem size, one line per config."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    drawn = False
    for tag, _, _, _, _ in CONFIGS:
        pts = [(r["gridpoints"], r["continuity"])
               for r in rows_by_cfg.get(tag, []) if r.get("continuity")]
        if not pts:
            continue
        drawn = True
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                label=DISPLAY[tag], **_STYLE[tag])
    if not drawn:
        plt.close(fig)
        return None
    _mark_reference_points(ax)
    _annotate_scaling(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    ax.set_ylabel("(Ocean continuity equation) wall time (s)")
    ax.set_title("Continuity solver vs problem size "
                 "(iturbo includes bridge)")
    ax.set_ylim(top=ax.get_ylim()[1] * 3.0)  # headroom for the scaling labels
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


def plot_kernels(rows_by_cfg, outpath):
    """The five ported PPM kernels vs problem size, all configurations.

    One panel per kernel; one line per config that has per-kernel timers
    (dev/turbo: MOM6 cpu_clock; iturbo amrex: AMReX TinyProfiler).
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    flat = axes.ravel()
    drawn = False
    for ax, (disp, _) in zip(flat, KERNELS):
        for tag, _, _, _, _ in CONFIGS:
            pts = [(r["gridpoints"], r["kernels"].get(disp))
                   for r in rows_by_cfg.get(tag, [])
                   if r["kernels"].get(disp)]
            if not pts:
                continue
            drawn = True
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    label=DISPLAY[tag], **_STYLE[tag])
        _mark_reference_points(ax, node_line=True, text=False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(disp, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("gridpoints")
        ax.set_ylabel("wall time (s)")
        ax.legend(fontsize=9)
    flat[len(KERNELS)].axis("off")
    if not drawn:
        plt.close(fig)
        return None
    fig.suptitle("Ported continuity PPM kernels, all configurations "
                 "(dev: cpu_clock; iturbo amrex: TinyProfiler inclusive)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


# The outer continuity kernels encompass all the reconstruction-side compute:
# their BL_PROFILE / cpu_clock regions nest (edge_thickness > reconstruction >
# limiter), so summing this outermost pair captures the full per-direction
# kernel cost without double-counting the inner levels.
KERNEL_ONLY = ("zonal_edge_thickness", "meridional_edge_thickness")


def _kernel_only_time(row):
    """Summed wall time of the outer (all-encompassing) continuity kernels for a
    row, or None if either is missing."""
    ts = [row["kernels"].get(k) for k in KERNEL_ONLY]
    return sum(ts) if all(t is not None for t in ts) else None


def plot_kernel_speedup(rows_by_cfg, outpath):
    """iturbo vs dev/turbo speedup of the continuity KERNELS ONLY -- compute,
    excluding the per-call host<->device bridge marshalling -- vs problem size.

    The kernel timers (BL_PROFILE in mom_continuity_ppm.cpp for iturbo, cpu_clock
    for dev/turbo) wrap only the kernel compute; the bridge's H2D/D2H copies are
    unmarked and so excluded here. Contrast the whole-model throughput/continuity
    plots, which DO include that marshalling."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    drawn = False
    for label, base_tag, cfg_tag in SPEEDUPS:
        base_by_i = {r["i"]: r for r in rows_by_cfg.get(base_tag, [])}
        pts = []
        for r in rows_by_cfg.get(cfg_tag, []):
            base = base_by_i.get(r["i"])
            if not base:
                continue
            kt, bt = _kernel_only_time(r), _kernel_only_time(base)
            if kt and bt:
                pts.append((r["gridpoints"], bt / kt))
        pts.sort()
        if not pts:
            continue
        drawn = True
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                label=f"{DISPLAY[cfg_tag]}\n   / {DISPLAY[base_tag]}",
                **_STYLE[cfg_tag])
    if not drawn:
        plt.close(fig)
        return None
    ax.axhline(1.0, ls="-", color="black", alpha=0.6, lw=1)
    _mark_reference_points(ax)
    _annotate_scaling(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    ax.set_ylabel("Kernel-only speedup vs dev/turbo (>1 = iturbo faster)")
    ax.set_title("iturbo vs dev/turbo: continuity kernels only "
                 "(excludes bridge)")
    _lo, _hi = ax.get_ylim()  # headroom for the scaling labels (linear axis)
    ax.set_ylim(top=_hi + 0.25 * (_hi - _lo))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


# --- tables -----------------------------------------------------------------

def fmt_int(n):
    return f"{n:,}" if n is not None else "-"


def fmt_sec(x):
    if x is None:
        return "-"
    return f"{x:.1f}" if x >= 100 else f"{x:.3f}"


def fmt_ratio(base, val):
    if base is None or val is None or val == 0:
        return "n/a"
    return f"{base / val:.2f}x"


def config_table(stacks):
    """The four configurations: plot label, executable origin, resources, kernel
    routing. The `plot label` column is the legend label used in every figure."""
    head = ("| config | abbrev | plot label | stack | resources "
            "| PPM kernel routing |\n"
            "|---|---|---|---|---|---|\n")
    body = ""
    for tag, abbrev, platform, _, stack in CONFIGS:
        res = "min(i, 128) ranks" if platform == "cpu" else "1 rank + 1 GPU"
        routing = ("AMReX (`*_MODE=AMREX`)" if "amrex" in tag else "Fortran")
        body += (f"| `{tag}` | {abbrev} | {DISPLAY[tag]} "
                 f"| `{stack}` (`{stacks.get(stack, '?')}`) "
                 f"| {res} | {routing} |\n")
    return head + body


def results_table(rows):
    """Per-config sweep results, repeats averaged (spread = min-max main loop)."""
    head = ("| i | ranks | NI x NJ | gridpoints | dt | runs | main loop (s) "
            "| spread (s) | s/step | throughput (cell-up/s) | continuity (s) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        body += (f"| {r['i']} | {r['nranks']} | {r['ni']}x{r['nj']} "
                 f"| {fmt_int(r['gridpoints'])} | {r['dt']} | {r['nruns']} "
                 f"| {r['main_loop']:.3f} "
                 f"| {r['main_loop_min']:.3f}-{r['main_loop_max']:.3f} "
                 f"| {r['sec_per_step']:.4f} | {r['throughput']:.3e} "
                 f"| {fmt_sec(r.get('continuity'))} |\n")
    return head + body


def headtohead_table(rows_by_cfg, platform):
    """Main-loop seconds for the platform's three configs side by side, per size,
    with each iturbo variant's speedup vs its dev/turbo baseline."""
    tags = [c[0] for c in CONFIGS if c[2] == platform]
    base_tag, variants = tags[0], tags[1:]
    by_i = {tag: {r["i"]: r for r in rows_by_cfg.get(tag, [])} for tag in tags}
    sizes = sorted(set().union(*[set(d) for d in by_i.values()]))
    if not sizes or not any(by_i[t] for t in tags):
        return None
    cols = [ABBREV[t] + " (s)" for t in tags] + \
           [ABBREV[v] + " speedup" for v in variants]
    head = ("| i | gridpoints | " + " | ".join(cols) + " |\n"
            "|---|---|" + "---|" * len(cols) + "\n")
    body = ""
    for i in sizes:
        some = next(by_i[t][i] for t in tags if i in by_i[t])
        cells = [str(i), fmt_int(some["gridpoints"])]
        loops = {t: by_i[t][i]["main_loop"] if i in by_i[t] else None for t in tags}
        cells += [fmt_sec(loops[t]) for t in tags]
        cells += [fmt_ratio(loops[base_tag], loops[v]) for v in variants]
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def kernel_snapshot_table(rows_by_cfg, snap_i):
    """The five kernels + aggregate clocks at one size, all configs side by side."""
    by_cfg = {tag: next((r for r in rows_by_cfg.get(tag, []) if r["i"] == snap_i),
                        None) for tag, *_ in CONFIGS}
    if not any(by_cfg.values()):
        return None
    tags = [c[0] for c in CONFIGS]
    head = ("| timer | " + " | ".join(ABBREV[t] for t in tags) + " |\n"
            "|---|" + "---|" * len(tags) + "\n")
    body = ""
    for disp, _ in KERNELS:
        cells = []
        for t in tags:
            r = by_cfg[t]
            v = r["kernels"].get(disp) if r else None
            src = r["kernel_source"].get(disp) if r else None
            cells.append(f"{fmt_sec(v)}" + (f" ({src})" if src else ""))
        body += f"| {disp} | " + " | ".join(cells) + " |\n"
    for label, key in (("continuity (mpp_clock)", "continuity"),
                       ("main loop (mpp_clock)", "main_loop")):
        cells = [fmt_sec(by_cfg[t][key]) if by_cfg[t] else "-" for t in tags]
        body += f"| {label} | " + " | ".join(cells) + " |\n"
    return head + body


def kernel_speedup_table(rows_by_cfg, snap_i):
    """Speedup vs the dev/turbo baseline on the same hardware, at one size."""
    by_cfg = {tag: next((r for r in rows_by_cfg.get(tag, []) if r["i"] == snap_i),
                        None) for tag, *_ in CONFIGS}

    def val(tag, kind, name):
        r = by_cfg.get(tag)
        if not r:
            return None
        return r["kernels"].get(name) if kind == "kernel" else r.get(name)

    rows = [(disp, "kernel", disp) for disp, _ in KERNELS]
    rows += [("continuity", "clock", "continuity"),
             ("main loop", "clock", "main_loop")]
    if not any(val(cfg, kind, name)
               for _, _, cfg in SPEEDUPS for _, kind, name in rows):
        return None
    head = ("| timer | " + " | ".join(lbl for lbl, _, _ in SPEEDUPS) + " |\n"
            "|---|" + "---|" * len(SPEEDUPS) + "\n")
    body = ""
    for disp, kind, name in rows:
        cells = [fmt_ratio(val(b, kind, name), val(c, kind, name))
                 for _, b, c in SPEEDUPS]
        body += f"| {disp} | " + " | ".join(cells) + " |\n"
    return head + body


def stats_check_table(run_dir, rows_by_cfg):
    """Whether ocean.stats files agree across configs (and repeats), per size.

    Byte-identity of the diagnostic checksum file is a cheap cross-config
    correctness signal; CPU and GPU groups are checked separately since their
    answers may legitimately differ in the last digits.
    """
    sizes = sorted({r["i"] for rows in rows_by_cfg.values() for r in rows})
    if not sizes:
        return None

    def digests(tag, i):
        out = set()
        r = 1
        while True:
            p = os.path.join(run_dir, f"{tag}_{i:03d}_run{r}.stats")
            if not os.path.isfile(p):
                break
            with open(p, "rb") as fh:
                out.add(hashlib.md5(fh.read()).hexdigest())
            r += 1
        return out

    head = ("| i | CPU configs | GPU configs |\n|---|---|---|\n")
    body = ""
    found = False
    for i in sizes:
        cells = []
        for platform in ("cpu", "gpu"):
            tags = [c[0] for c in CONFIGS if c[2] == platform]
            per_tag = {t: digests(t, i) for t in tags}
            all_d = set().union(*per_tag.values())
            nconfigs = len([t for t in tags if per_tag[t]])
            if not all_d:
                cells.append("no .stats files")
                continue
            found = True
            if len(all_d) == 1:
                cells.append(f"identical ({nconfigs} configs)")
            else:
                cells.append(f"**differ** ({len(all_d)} distinct, "
                             f"{nconfigs} configs)")
        body += f"| {i} | " + " | ".join(cells) + " |\n"
    return head + body if found else None


def failures_table(failures):
    """Runs that produced no Main loop timer, so they did not complete."""
    if not failures:
        return None
    head = ("| config | i | run | NI x NJ | gridpoints | log | cause (from stderr) |\n"
            "|---|---|---|---|---|---|---|\n")
    body = ""
    for f in failures:
        reason = f.get("reason") or "_(no stderr captured)_"
        body += (f"| `{f['config']}` | {f['i']} | {f['run']} | {f['ni']}x{f['nj']} "
                 f"| {fmt_int(f['gridpoints'])} | `{f['fname']}` | {reason} |\n")
    return head + body


# --- csv --------------------------------------------------------------------

def write_csv(rows_by_cfg, nsteps, path):
    """One row per (config, size, repeat) -- raw, unaveraged measurements."""
    kcols = [f"kernel_{disp}" for disp, _ in KERNELS]
    scols = [f"kernel_{disp}_source" for disp, _ in KERNELS]
    cols = (["config", "platform", "i", "run", "ni", "nj", "nk", "nranks",
             "gridpoints", "gridpoints_per_rank", "dt", "nsteps", "main_loop",
             "total_runtime", "init", "termination", "sec_per_step",
             "throughput", "continuity", "continuity_frac"] + kcols + scols)
    out = []
    for tag, _, platform, _, _ in CONFIGS:
        for r in rows_by_cfg.get(tag, []):
            for p in r["raw"]:
                rec = {
                    "config": tag, "platform": platform, "i": r["i"],
                    "run": p["run"], "ni": r["ni"], "nj": r["nj"], "nk": r["nk"],
                    "nranks": r["nranks"], "gridpoints": r["gridpoints"],
                    "gridpoints_per_rank": r["gridpoints_per_rank"],
                    "dt": r["dt"], "nsteps": nsteps,
                    "main_loop": p["main_loop"],
                    "total_runtime": p["total_runtime"], "init": p["init"],
                    "termination": p["termination"],
                    "sec_per_step": p["main_loop"] / nsteps,
                    "throughput": r["gridpoints"] * nsteps / p["main_loop"],
                    "continuity": p["continuity"],
                    "continuity_frac": (p["continuity"] / p["main_loop"]
                                        if p["continuity"] else None),
                }
                for disp, _ in KERNELS:
                    rec[f"kernel_{disp}"] = p["kernels"].get(disp)
                    rec[f"kernel_{disp}_source"] = p["kernel_source"].get(disp)
                out.append(rec)
    if not out:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec in out:
            w.writerow(rec)


# --- report -----------------------------------------------------------------

def build_report(rows_by_cfg, failures, plots, snap_i, nsteps, title, stacks,
                 run_dir, prov=None):
    """Assemble the facts-only Markdown report (see gen_report.py for the
    facts-vs-commentary rationale)."""
    block_str = f"{BLOCK}x{BLOCK}x{NK}"
    L = []
    L.append(f"# {title}\n")
    if prov is not None:
        L.append(render_stamp(prov))

    L.append("## Intent\n")
    L.append(
        "Compare four MOM6 `double_gyre` configurations -- "
        "{dev/turbo, iturbo-AMReX} x {CPU, GPU} -- across the standard "
        "problem-size sweep on Derecho. Each pairs an executable from a "
        "specific turbo-stack checkout with, for the AMReX variants, the six "
        "`*_MODE=AMREX` env vars that route the ported continuity PPM kernels "
        "through C++/AMReX.\n")
    L.append(config_table(stacks))
    L.append("\n<!-- commentary: key-finding -->\n")

    L.append("## Methodology\n")
    L.append(
        f"Each run advances exactly **{nsteps} dynamic steps** "
        "(`TIMEUNIT = dt`, `DAYMAX = 150`), so wall-clock is comparable across "
        f"sizes and configs. Job-size index `i` is a near-square layout of `i` "
        f"{BLOCK}x{BLOCK} blocks at NK={NK}.\n\n"
        "- **CPU configs**: weak scaling -- ranks grow with `i` at a constant "
        f"{block_str} gridpoints/rank up to the {CPU_PER_NODE}-rank node cap, "
        f"then stay at {CPU_PER_NODE} while per-rank work grows.\n"
        "- **GPU configs**: single-device scan (1 rank, 1 A100, 1x1).\n\n"
        "Each (config, size) point runs **N times** (`runs` columns); timers "
        "are averaged, and `spread` is the min-max of the main-loop timer over "
        "repeats.\n\n"
        "Aggregate timers are the cross-PE mean (\"tavg\") of FMS `mpp_clock` "
        "rows (e.g. `Main loop`, `(Ocean continuity equation)`). Per-kernel "
        "timers come from two sources -- dev/turbo: MOM6 `cpu_clock`; "
        "iturbo-AMReX: AMReX TinyProfiler INCLUSIVE (run with "
        "`tiny_profiler.device_synchronize_around_region=1`) -- each falling "
        "back to the other, with every kernel cell annotated by source. Both "
        "are inclusive wall-clock (launch + execution + sync), so directly "
        "comparable. (This report uses mpp_clock tavg; the single-size harness "
        "used tmax -- identical for single-rank GPU runs.) See "
        "`docs/COMPARE_SWEEP.md`.\n")
    L.append("<!-- commentary: methodology -->\n")

    if plots.get("throughput"):
        L.append("## Throughput vs problem size (whole model)\n")
        L.append(f"![Throughput]({os.path.basename(plots['throughput'])})\n")
        L.append(
            "Cell-updates/s vs problem size, all four configs. **Color** = "
            "platform (CPU blue, GPU orange); **marker/linestyle** = build "
            "variant (dev/turbo solid squares, iturbo-AMReX dashed circles). "
            "Dotted red vertical: i=128 (13.1M gridpoints), where the CPU run "
            "fills a full Derecho CPU node (128 cores) -- a clean "
            "128-cores-vs-1-GPU comparison. Dashed gray: the 19.4M production "
            "point. This is a **whole-model** rate (total cell-updates / "
            "main-loop time), so for iturbo it includes the per-call "
            "host<->device bridge marshalling, not just the kernels -- see the "
            "kernel-only throughput below.\n")
        L.append("<!-- commentary: throughput -->\n")

    if plots.get("kernel_throughput"):
        L.append("## Throughput vs problem size: compute kernels only\n")
        L.append(f"![Kernel throughput]({os.path.basename(plots['kernel_throughput'])})\n")
        L.append(
            "The same cell-updates/s metric from the continuity kernel "
            "**compute** time alone (outer zonal + meridional `edge_thickness`), "
            "excluding the AMReX bridge's host<->device marshalling; for "
            "iturbo, its gap from the whole-model throughput above is the bridge "
            "overhead. Encoding and verticals as above.\n")
        L.append(
            "\"Compute\" here means kernel **launch + on-device execution + "
            "sync**, not pure arithmetic -- both configs are timed on this same "
            "basis (`device_synchronize_around_region=1`; the dev/turbo `do "
            "concurrent` launches are host-synchronous), and only the bridge "
            "transfers are excluded, so the comparison is fair.\n")
        L.append("<!-- commentary: kernel-throughput -->\n")

    if plots.get("kernel_speedup"):
        L.append("## iturbo vs dev/turbo speedup: kernels only\n")
        L.append(f"![Kernel speedup]({os.path.basename(plots['kernel_speedup'])})\n")
        L.append(
            "The same ratio restricted to the **continuity kernel compute** "
            "(outer `zonal_edge_thickness` + `meridional_edge_thickness`; "
            "`BL_PROFILE` for iturbo, `cpu_clock` for dev/turbo), excluding the "
            "bridge's H2D/D2H copies -- isolating the port's compute from its "
            "integration overhead. Its gap from the whole-model "
            "throughput/continuity curves is the bridge tax. CPU and GPU pairs "
            "are shown where timers exist; encoding and verticals as above.\n")
        L.append("<!-- commentary: kernel-speedup -->\n")

    for platform, label in (("cpu", "CPU"), ("gpu", "GPU")):
        t = headtohead_table(rows_by_cfg, platform)
        if t:
            L.append(f"## Head-to-head: {label} configs\n")
            L.append(
                f"Main-loop seconds for the three {label} configurations at "
                "each size, with each iturbo variant's speedup vs dev/turbo "
                "(>1 = iturbo faster). Missing cells are sizes that config did "
                "not complete.\n")
            L.append(t)
            L.append("")
    L.append("<!-- commentary: head-to-head -->\n")

    if plots.get("continuity"):
        L.append("## Continuity solver\n")
        L.append(f"![Continuity]({os.path.basename(plots['continuity'])})\n")
        L.append(
            "The `(Ocean continuity equation)` mpp_clock timer vs problem size, "
            "all configs -- the routine whose PPM kernels the AMReX port "
            "replaces, timed end-to-end. For iturbo this folds in the "
            "host<->device transfers and runtime overhead, not just kernels "
            "(hence \"includes bridge\"). Verticals as above.\n")
        L.append("<!-- commentary: continuity -->\n")

    if plots.get("kernels"):
        L.append("## Ported PPM kernels\n")
        L.append(f"![Kernels]({os.path.basename(plots['kernels'])})\n")
        L.append(
            "Wall-clock of the five ported PPM kernels vs problem size, all "
            "configs (timer sources per Methodology; launch + execution + sync, "
            "bridge transfers excluded). The kernels nest (`edge_thickness` > "
            "`reconstruction` > `limiter`), so rows are inclusive, not "
            "additive.\n")
        L.append("<!-- commentary: kernels -->\n")

    if snap_i is not None:
        kt = kernel_snapshot_table(rows_by_cfg, snap_i)
        st = kernel_speedup_table(rows_by_cfg, snap_i)
        if kt or st:
            L.append(f"## Kernel snapshot at i={snap_i}\n")
            L.append(
                f"All configurations side by side at job size i={snap_i} "
                "(the largest size completed by every config). Seconds, "
                "averaged over repeats; each kernel cell notes its timer "
                "source (`mom6` cpu_clock or `tiny` TinyProfiler inclusive).\n")
            if kt:
                L.append(kt)
            if st:
                L.append("\nSpeedup vs the dev/turbo baseline on the same "
                         "hardware (>1 = iturbo variant faster):\n")
                L.append(st)
            L.append("\n<!-- commentary: kernel-snapshot -->\n")

    stats_t = stats_check_table(run_dir, rows_by_cfg)
    if stats_t:
        L.append("## ocean.stats cross-check\n")
        L.append(
            "Byte-identity of the `ocean.stats` diagnostic file across the "
            "configs (and repeats) of each platform group, per size -- a cheap "
            "signal that all variants computed the same physics. CPU and GPU "
            "groups are checked separately.\n")
        L.append(stats_t)
        L.append("\n<!-- commentary: ocean-stats -->\n")

    ft = failures_table(failures)
    if ft:
        L.append("## Failed / missing runs\n")
        L.append(
            "These runs produced no FMS `Main loop` timer, so they did not "
            "complete and are excluded from the plots and tables above (their "
            "repeats that did complete are still averaged). The `cause` column "
            "is the failing line from the run's stderr.\n")
        L.append(ft)
        L.append("\n<!-- commentary: failures -->\n")

    L.append("## Results by configuration\n")
    for tag, abbrev, _, _, _ in CONFIGS:
        rows = rows_by_cfg.get(tag)
        L.append(f"### {tag} ({abbrev})\n")
        L.append(results_table(rows) if rows else "_No completed runs found._\n")

    if prov is not None:
        L.append(render_provenance_multi(prov, include_stamp=False))

    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="directory with <config>_<i>_run<r>.out logs")
    ap.add_argument("--stack", action="append", default=[], metavar="NAME=PATH",
                    help="turbo-stack checkout for provenance; repeatable. "
                    "Names: " + ", ".join(DEFAULT_STACKS) +
                    " (defaults: the standard /glade/work checkouts)")
    ap.add_argument("--configs", nargs="*", default=[c[0] for c in CONFIGS],
                    help="subset of configs to include (default: all four)")
    ap.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR,
                    help="parent directory for timestamped report dirs "
                    "(default: ../reports)")
    ap.add_argument("--label", default="compare-sweep",
                    help="trailing label for the report dir name "
                    "(default: compare-sweep); dir is <date-time>-<label>")
    ap.add_argument("--outdir", help="explicit output directory; overrides the "
                    "timestamped --reports-dir/--label naming")
    ap.add_argument("--nsteps", type=int, default=NSTEPS_DEFAULT,
                    help=f"dynamic steps per run (default {NSTEPS_DEFAULT})")
    ap.add_argument("--snapshot-i", type=int,
                    help="job-size index for the kernel snapshot tables "
                    "(default: largest size completed by every config)")
    ap.add_argument("--title",
                    default="MOM6 double_gyre four-config comparison sweep")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip PNG generation (no matplotlib needed)")
    ap.add_argument("--note", help="free-form note added to the provenance block")
    ap.add_argument("--date", help="override the generated timestamp text in "
                    "the provenance block; defaults to now (YYYY-MM-DD HH:MM:SS)")
    args = ap.parse_args()

    stacks = dict(DEFAULT_STACKS)
    for spec in args.stack:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--stack expects NAME=PATH, got: {spec}")
        stacks[name] = path

    now = datetime.datetime.now()
    gen_time = args.date or now.isoformat(sep=" ", timespec="seconds")

    rows_by_cfg, failures = {}, []
    for tag, _, platform, prefer_tiny, _ in CONFIGS:
        if tag not in args.configs:
            continue
        rows, fails = collect_compare(args.run_dir, tag, platform, prefer_tiny)
        add_throughput(rows, args.nsteps)
        rows_by_cfg[tag] = rows
        failures += fails
        print(f"  {tag}: {len(rows)} size(s), "
              f"{sum(r['nruns'] for r in rows)} completed run(s), "
              f"{len(fails)} failed.")

    # Snapshot size: the largest i completed by every config that has any runs.
    populated = [set(r["i"] for r in rows) for rows in rows_by_cfg.values() if rows]
    shared = sorted(set.intersection(*populated)) if populated else []
    snap_i = args.snapshot_i if args.snapshot_i is not None else \
        (max(shared) if shared else None)

    prov = gather_provenance_multi(stacks, args.note, gen_time)

    outdir = resolve_outdir(args, now)
    os.makedirs(outdir, exist_ok=True)
    write_csv(rows_by_cfg, args.nsteps, os.path.join(outdir, "results.csv"))
    with open(os.path.join(outdir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    plots = {}
    if not args.no_plots:
        plots["throughput"] = plot_throughput(
            rows_by_cfg, os.path.join(outdir, "throughput.png"))
        plots["kernel_throughput"] = plot_kernel_throughput(
            rows_by_cfg, os.path.join(outdir, "kernel_throughput.png"), args.nsteps)
        plots["kernel_speedup"] = plot_kernel_speedup(
            rows_by_cfg, os.path.join(outdir, "kernel_speedup.png"))
        plots["continuity"] = plot_continuity(
            rows_by_cfg, os.path.join(outdir, "continuity.png"))
        plots["kernels"] = plot_kernels(
            rows_by_cfg, os.path.join(outdir, "kernels.png"))

    report = build_report(rows_by_cfg, failures, plots, snap_i, args.nsteps,
                          args.title, stacks, args.run_dir, prov=prov)
    with open(os.path.join(outdir, "report.md"), "w") as fh:
        fh.write(report)
    print(f"Wrote report to {outdir}/")
    print("  report.md, results.csv, provenance.json")
    for p in plots.values():
        if p:
            print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
