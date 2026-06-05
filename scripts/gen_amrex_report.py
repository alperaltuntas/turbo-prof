#!/usr/bin/env python3
"""Generate a Markdown report on the AMReX-ported continuity PPM kernels.

This is the **AMReX continuity** report type -- distinct from the whole-model
scaling report (gen_report.py). It quantifies the cost of the continuity PPM
sub-kernels that were ported to C++/AMReX, and how that cost splits between actual
GPU *kernel* compute and the *data movement* the bridge does around each call (both
the on-device data-repacking kernels and the host<->device PCIe copies). The FMS
mpp_clock timer folds all of that together; only a CUDA profile (Nsight Systems)
separates them, which is the point of this report.

Inputs (produced by run-profile.sh):
  prof_<mode>_<i>.nsys-rep   Nsight Systems trace
  prof_<mode>_<i>.out        MOM6 stdout incl. the FMS mpp_clock table
where <mode> in {fortran, amrex} and <i> is the job-size index. The two modes run
the SAME executable; AMREX sets six *_MODE env vars so the ported PPM kernels take
the AMReX/CUDA path instead of the Fortran (OpenMP-offload) path.

Method (see PROFILING_DECISIONS.md):
  * Kernel time -- `nsys stats --report cuda_gpu_kern_sum:mangled`. nsys's demangled
    view abbreviates template args (T2/T3), hiding the function inside the lambda, so
    we take the MANGLED names and demangle with c++filt. Classify by name:
      - MOM::PPM_reconstruction_x/y, MOM::ppm_limit_pos/cw84,
        MOM::{zonal,meridional}_edge_thickness   -> ported PPM COMPUTE
      - turbotmp::copy_FortranHost_to_array4,
        turbotmp::copy_array4_to_FortranHost     -> bridge DATA REPACKING (layout convert)
      - other amrex::/turbotmp::                 -> AMReX infrastructure
      - everything else (nvfortran)              -> whole-model OpenMP offload
  * PCIe copies -- `cuda_gpu_mem_time_sum`/`_mem_size_sum` give the device-side H2D/D2H
    totals (bridge + OpenMP lumped). `cuda_api_sum` separates them by API: the AMReX
    bridge issues `cudaMemcpyAsync` (runtime API); the OpenMP offload uses
    `cuMemcpy{2D,DtoH,HtoD}Async_v2` (driver API).

IMPORTANT: re-processing the .nsys-rep needs the SAME (or newer) nsys that recorded
it -- here the nvhpc-25.9 nsys (>=2025.5). The cuda/12.9.0 nsys (2025.1) is older and
exports an empty database. Load nvhpc after cuda, or set NSYS=/path/to/nsys.

The report is **facts only**; interpretation is added by the report-commentary skill,
which fills the `<!-- commentary: NAME -->` anchors. See docs/REPORTING.md.

Usage:
    python3 gen_amrex_report.py --prof-dir DIR [--stack-dir DIR] [--outdir DIR] \
        [--nsteps 20] [--title "..."] [--no-plots]

ENVIRONMENT: keep the nvhpc/25.9 modules loaded so `nsys` resolves to the >=2025.5
binary that can read the trace. You can launch with the plain (nvhpc) python -- for
the figures it auto-re-execs under a matplotlib-capable python (the `npl` conda env,
or $MATPLOTLIB_PYTHON), inheriting PATH so `nsys` still resolves. Pass --no-plots to
skip figures (the facts tables render either way).
"""

import argparse
import csv
import datetime
import glob
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turboprof.parsing import parse_run
from turboprof.provenance import gather_provenance, render_provenance, render_stamp
from turboprof.reporting import DEFAULT_REPORTS_DIR, resolve_outdir

NSTEPS_DEFAULT = 20
MODES = ("fortran", "amrex")
NSYS = os.environ.get("NSYS", "nsys")

# Ported PPM compute kernels, keyed by the demangled-name substring (MOM::<fn>).
# reconstruction_x/y each appear as two ParallelFor lambdas (slope + edge); both map
# to the same key and are summed. edge_thickness routines delegate to reconstruction,
# so they rarely appear as their own kernels; cw84 only with a monotonic limiter.
PORTED_KERNELS = [
    ("PPM_reconstruction_x",      "PPM reconstruction x (zonal)"),
    ("PPM_reconstruction_y",      "PPM reconstruction y (meridional)"),
    ("ppm_limit_pos",             "PPM limiter: positive-definite"),
    ("ppm_limit_cw84",            "PPM limiter: CW84 monotonic"),
    ("zonal_edge_thickness",      "zonal edge thickness"),
    ("meridional_edge_thickness", "meridional edge thickness"),
]
# Static facts about the continuity PPM dispatch (from the MOM6/TIM source, not
# from any run): three of the six entry points launch their own kernel only on a
# code path that a given run may not take, so a zero is structural rather than a
# missing measurement. Surfaced as a Note column. These describe the code, not the
# experiment -- interpretation of the run stays in the commentary anchors.
PORTED_KERNEL_NOTES = {
    "ppm_limit_cw84":
        "limiter selected by MONOTONIC_CONTINUITY; the alternative is the "
        "positive-definite limiter",
    "zonal_edge_thickness":
        "wrapper over PPM_reconstruction_x; launches its own kernel only on the "
        "1st-order-upwind path",
    "meridional_edge_thickness":
        "wrapper over PPM_reconstruction_y; launches its own kernel only on the "
        "1st-order-upwind path",
}
# Bridge data-repacking kernels (host Fortran layout <-> AMReX Array4 layout).
REPACK_KERNELS = [
    ("copy_FortranHost_to_array4", "host -> Array4 pack (H2D side)"),
    ("copy_array4_to_FortranHost", "Array4 -> host unpack (D2H side)"),
]
# API names: which memcpy entry points belong to the AMReX bridge vs OpenMP offload.
BRIDGE_COPY_APIS = ("cudaMemcpyAsync", "cudaMemcpy")
OPENMP_COPY_APIS = ("cuMemcpy2DAsync_v2", "cuMemcpyDtoHAsync_v2",
                    "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoDAsync_v2")


# --- nsys CSV extraction ----------------------------------------------------

def _report_filebase(report):
    """nsys turns 'cuda_gpu_kern_sum:mangled' into the file suffix '..._mangled'."""
    return report.replace(":", "_")


def _nsys_csv(rep_path, reports, workdir):
    """Run `nsys stats` on a .nsys-rep; return {report: [row dicts]} via -o CSVs."""
    base = os.path.join(workdir, os.path.splitext(os.path.basename(rep_path))[0])
    cmd = [NSYS, "stats", "--force-export=true", "--format", "csv", "-o", base]
    for r in reports:
        cmd += ["--report", r]
    cmd.append(rep_path)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = {}
    for r in reports:
        path = f"{base}_{_report_filebase(r)}.csv"
        rows = []
        if os.path.exists(path):
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    rows.append({(k or "").strip(): (v or "").strip()
                                 for k, v in row.items()})
        out[r] = rows
    if not any(out[r] for r in reports):
        sys.stderr.write(
            f"WARNING: nsys produced no rows for {os.path.basename(rep_path)}.\n"
            f"  Likely an nsys version mismatch (need the recorder's nsys, e.g.\n"
            f"  nvhpc-25.9 >=2025.5). Set NSYS=/path/to/nsys. nsys said:\n"
            + "  " + (proc.stdout.decode(errors="replace")[-500:] if proc.stdout else "")
            + "\n")
    return out


def _demangle(names):
    """Batch-demangle C++ symbols with c++filt; returns list aligned to input."""
    if not names:
        return []
    try:
        p = subprocess.run(["c++filt"], input="\n".join(names).encode(),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out = p.stdout.decode(errors="replace").splitlines()
        if len(out) == len(names):
            return out
    except FileNotFoundError:
        pass
    return names  # c++filt missing: fall back to raw (classification may degrade)


def _num(s):
    s = (s or "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def _col(row, *names):
    norm = {re.sub(r"\s+", "", k).lower(): v for k, v in row.items()}
    for n in names:
        v = norm.get(re.sub(r"\s+", "", n).lower())
        if v is not None:
            return v
    return ""


# --- kernel classification --------------------------------------------------

def classify_kernel(name):
    """Bucket by demangled name: 'ported' | 'repack' | 'amrex' | 'openmp'."""
    if "MOM::" in name:
        return "ported"
    if any(k in name for k, _ in REPACK_KERNELS):
        return "repack"
    if "amrex::" in name or "turbotmp::" in name:
        return "amrex"
    return "openmp"


def _key_for(name, table):
    for key, _ in table:
        if key.lower() in name.lower():
            return key
    return None


def summarize_kernels(kern_rows):
    """Aggregate cuda_gpu_kern_sum:mangled rows (names demangled) into buckets."""
    names = [_col(r, "Name") for r in kern_rows]
    demangled = _demangle(names)
    buckets = {b: 0.0 for b in ("ported", "repack", "amrex", "openmp")}
    per_ported, per_repack = {}, {}
    for row, name in zip(kern_rows, demangled):
        if not name:
            continue
        t = _num(_col(row, "Total Time (ns)", "Total Time"))
        n = _num(_col(row, "Instances", "Count"))
        b = classify_kernel(name)
        buckets[b] += t
        if b == "ported":
            k = _key_for(name, PORTED_KERNELS)
            if k:
                d = per_ported.setdefault(k, {"ns": 0.0, "inst": 0.0})
                d["ns"] += t; d["inst"] += n
        elif b == "repack":
            k = _key_for(name, REPACK_KERNELS)
            if k:
                d = per_repack.setdefault(k, {"ns": 0.0, "inst": 0.0})
                d["ns"] += t; d["inst"] += n
    return {"buckets": buckets, "per_ported": per_ported, "per_repack": per_repack}


def summarize_memops(time_rows, size_rows):
    def direction(op):
        o = op.lower()
        if "host-to-device" in o or "htod" in o:
            return "htod"
        if "device-to-host" in o or "dtoh" in o:
            return "dtoh"
        return "other"
    agg = {d: {"ns": 0.0, "mb": 0.0} for d in ("htod", "dtoh", "other")}
    for row in time_rows:
        agg[direction(_col(row, "Operation"))]["ns"] += \
            _num(_col(row, "Total Time (ns)", "Total Time"))
    for row in size_rows:
        agg[direction(_col(row, "Operation"))]["mb"] += \
            _num(_col(row, "Total (MB)", "Total"))
    return agg


def summarize_api_copies(api_rows):
    """Split memcpy API time into bridge (cudaMemcpyAsync) vs OpenMP (cuMemcpy*)."""
    agg = {"bridge": {"ns": 0.0, "calls": 0.0}, "openmp": {"ns": 0.0, "calls": 0.0}}
    for row in api_rows:
        nm = _col(row, "Name")
        t = _num(_col(row, "Total Time (ns)", "Total Time"))
        c = _num(_col(row, "Num Calls", "Count", "Instances"))
        if nm in BRIDGE_COPY_APIS:
            agg["bridge"]["ns"] += t; agg["bridge"]["calls"] += c
        elif nm in OPENMP_COPY_APIS:
            agg["openmp"]["ns"] += t; agg["openmp"]["calls"] += c
    return agg


# --- collection -------------------------------------------------------------

def collect(prof_dir, workdir):
    reports = ["cuda_gpu_kern_sum:mangled", "cuda_gpu_mem_time_sum",
               "cuda_gpu_mem_size_sum", "cuda_api_sum"]
    runs = {}
    for rep in sorted(glob.glob(os.path.join(prof_dir, "prof_*.nsys-rep"))):
        m = re.search(r"prof_(fortran|amrex)_(\d+)\.nsys-rep$", os.path.basename(rep))
        if not m:
            continue
        mode, i = m.group(1), int(m.group(2))
        csvs = _nsys_csv(rep, reports, workdir)
        out_path = os.path.join(prof_dir, f"prof_{mode}_{i:03d}.out")
        runs[(mode, i)] = {
            "mode": mode, "i": i,
            "kern": summarize_kernels(csvs["cuda_gpu_kern_sum:mangled"]),
            "mem": summarize_memops(csvs["cuda_gpu_mem_time_sum"],
                                    csvs["cuda_gpu_mem_size_sum"]),
            "api": summarize_api_copies(csvs["cuda_api_sum"]),
            "timers": parse_run(out_path) if os.path.exists(out_path) else None,
            "completed": bool(parse_run(out_path)) if os.path.exists(out_path) else False,
        }
    return runs


# --- formatting -------------------------------------------------------------

def fmt_ms(ns):
    return f"{ns / 1e6:,.2f}"


def sizes_in(runs):
    return sorted({i for (_, i) in runs})


def grid_label(timers):
    if timers and timers.get("niglobal") and timers.get("njglobal"):
        return f"{timers['niglobal']}x{timers['njglobal']}x100"
    return "?"


def _amrex_runs(runs):
    """(i, run) for AMREX cells that produced GPU kernel data, sorted by size."""
    out = []
    for i in sizes_in(runs):
        r = runs.get(("amrex", i))
        if r and sum(r["kern"]["buckets"].values()) > 0:
            out.append((i, r))
    return out


def gridpoints(timers):
    """Total gridpoints NI*NJ*100 from a parsed .out, or None."""
    if timers and timers.get("niglobal") and timers.get("njglobal"):
        return timers["niglobal"] * timers["njglobal"] * 100
    return None


# --- plots ------------------------------------------------------------------
# Lazy matplotlib import (as in gen_report.py) so the facts tables still work in
# an env without it; each plotter returns its path, or None when it can't render.

def plot_compute_vs_movement(runs, outpath):
    """Stacked GPU-time bars for the ported continuity piece, one bar per size.

    Splits the per-call cost into PPM compute (the math) vs bridge data repacking
    (the host<->Array4 repack kernels) -- the headline of this report. The
    OpenMP whole-model kernels are excluded; this is the continuity piece only.
    """
    import matplotlib.pyplot as plt
    rows = _amrex_runs(runs)
    if not rows:
        return None
    labels = [grid_label(r["timers"]) for _, r in rows]
    compute = [r["kern"]["buckets"]["ported"] / 1e6 for _, r in rows]
    repack = [r["kern"]["buckets"]["repack"] / 1e6 for _, r in rows]
    x = range(len(rows))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, compute, color="C0", label="PPM compute (math)")
    ax.bar(x, repack, bottom=compute, color="C3",
           label="data repacking (host array <-> AMReX Array4 layout)")
    for xi, (c, m) in enumerate(zip(compute, repack)):
        if c:
            ax.text(xi, c + m, f"{m/c:.2f}x", ha="center", va="bottom",
                    fontsize=9, color="dimgray")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("GPU kernel time over run (ms)")
    ax.set_xlabel("Problem size (NI x NJ x 100)")
    ax.set_title("Ported continuity (AMREX mode): compute vs. data-repacking kernels")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.margins(y=0.12)  # headroom so the ratio labels clear the top
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


def plot_continuity_headtohead(runs, outpath):
    """FORTRAN vs AMReX continuity (mpp_clock) wall time vs problem size.

    The end-to-end comparison: the same solver, OpenMP Fortran path vs AMReX/CUDA
    path, matched by problem size. Needs both modes' timers.
    """
    import matplotlib.pyplot as plt
    fpts, apts = [], []
    for i in sizes_in(runs):
        rf, ra = runs.get(("fortran", i)), runs.get(("amrex", i))
        if rf and rf["timers"] and rf["timers"].get("continuity") is not None \
                and gridpoints(rf["timers"]):
            fpts.append((gridpoints(rf["timers"]), rf["timers"]["continuity"]))
        if ra and ra["timers"] and ra["timers"].get("continuity") is not None \
                and gridpoints(ra["timers"]):
            apts.append((gridpoints(ra["timers"]), ra["timers"]["continuity"]))
    if not fpts and not apts:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    if fpts:
        fpts.sort()
        ax.plot([p[0] for p in fpts], [p[1] for p in fpts], "s-", color="C0",
                label="FORTRAN (OpenMP offload)")
    if apts:
        apts.sort()
        ax.plot([p[0] for p in apts], [p[1] for p in apts], "o-", color="C1",
                label="AMREX (C++/CUDA bridge)")
    # speedup annotation where both exist
    fmap = dict(fpts)
    for gp, at in apts:
        if gp in fmap and at:
            ax.annotate(f"{fmap[gp]/at:.2f}x", (gp, at), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=8, color="C2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    ax.set_ylabel("Continuity solver wall time (s, mpp_clock tavg)")
    ax.set_title("Continuity solver: AMReX vs OpenMP-Fortran")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


def plot_ported_kernels(runs, outpath):
    """Per-kernel GPU time of the ported PPM kernels, grouped bars by size."""
    import matplotlib.pyplot as plt
    import numpy as np
    rows = _amrex_runs(runs)
    if not rows:
        return None
    present = [(k, lbl) for k, lbl in PORTED_KERNELS
               if any(r["kern"]["per_ported"].get(k) for _, r in rows)]
    if not present:
        return None
    labels = [grid_label(r["timers"]) for _, r in rows]
    x = np.arange(len(rows))
    w = 0.8 / max(len(present), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, (k, lbl) in enumerate(present):
        vals = [(r["kern"]["per_ported"].get(k, {}).get("ns", 0.0)) / 1e6
                for _, r in rows]
        ax.bar(x + j * w, vals, w, label=lbl)
    ax.set_xticks(x + w * (len(present) - 1) / 2)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("GPU compute time over run (ms)")
    ax.set_xlabel("Problem size (NI x NJ x 100)")
    ax.set_title("Ported PPM compute kernels (AMREX mode)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


def kernel_category_table(runs):
    L = ["| Size | PPM compute (ms) | Data repacking (ms) | AMReX infra (ms) | OpenMP whole-model (ms) |",
         "|---|--:|--:|--:|--:|"]
    for i, r in _amrex_runs(runs):
        b = r["kern"]["buckets"]
        L.append(f"| {grid_label(r['timers'])} | {fmt_ms(b['ported'])} | "
                 f"{fmt_ms(b['repack'])} | {fmt_ms(b['amrex'])} | {fmt_ms(b['openmp'])} |")
    return "\n".join(L)


def ported_breakdown_table(runs):
    L = ["A `-`/`0` row is expected for the kernels noted below, not a missing "
         "measurement.\n",
         "| Size | Ported PPM kernel | GPU compute time (ms) | Launches | Note |",
         "|---|---|--:|--:|---|"]
    for i, r in _amrex_runs(runs):
        per = r["kern"]["per_ported"]
        g = grid_label(r["timers"])
        for key, label in PORTED_KERNELS:
            d = per.get(key)
            note = PORTED_KERNEL_NOTES.get(key, "")
            if d:
                L.append(f"| {g} | {label} | {fmt_ms(d['ns'])} | {int(d['inst'])} | {note} |")
            else:
                L.append(f"| {g} | {label} | - | 0 | {note} |")
    return "\n".join(L)


def repacking_table(runs):
    L = ["| Size | Bridge data-repacking kernel | GPU time (ms) | Launches |",
         "|---|---|--:|--:|"]
    for i, r in _amrex_runs(runs):
        per = r["kern"]["per_repack"]
        g = grid_label(r["timers"])
        for key, label in REPACK_KERNELS:
            d = per.get(key)
            if d:
                L.append(f"| {g} | {label} | {fmt_ms(d['ns'])} | {int(d['inst'])} |")
            else:
                L.append(f"| {g} | {label} | - | 0 |")
    return "\n".join(L)


def compute_vs_repacking_table(runs):
    L = ["| Size | PPM compute (ms) | Data repacking kernels (ms) | Repacking / compute |",
         "|---|--:|--:|--:|"]
    for i, r in _amrex_runs(runs):
        b = r["kern"]["buckets"]
        comp, marsh = b["ported"], b["repack"]
        ratio = (marsh / comp) if comp else 0.0
        L.append(f"| {grid_label(r['timers'])} | {fmt_ms(comp)} | {fmt_ms(marsh)} | {ratio:.2f}x |")
    return "\n".join(L)


def pcie_copy_table(runs):
    L = ["| Size | H2D total (GB) | D2H total (GB) | H2D time (ms) | D2H time (ms) | Bridge memcpy API (ms, calls) | OpenMP memcpy API (ms, calls) |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for i, r in _amrex_runs(runs):
        mem, api = r["mem"], r["api"]
        L.append(
            f"| {grid_label(r['timers'])} | {mem['htod']['mb']/1024:,.1f} | "
            f"{mem['dtoh']['mb']/1024:,.1f} | {fmt_ms(mem['htod']['ns'])} | "
            f"{fmt_ms(mem['dtoh']['ns'])} | {fmt_ms(api['bridge']['ns'])} "
            f"({int(api['bridge']['calls'])}) | {fmt_ms(api['openmp']['ns'])} "
            f"({int(api['openmp']['calls'])}) |")
    return "\n".join(L)


def folded_timer_table(runs):
    L = ["| Size | Mode | Status | mpp_clock continuity tavg (s) | Main loop tavg (s) |",
         "|---|---|---|--:|--:|"]
    for i in sizes_in(runs):
        for mode in MODES:
            r = runs.get((mode, i))
            if not r:
                continue
            t, g = r["timers"], grid_label(r["timers"])
            if not r["completed"] or not t:
                L.append(f"| {g if g!='?' else i} | {mode.upper()} | did not complete | - | - |")
                continue
            cont = t.get("continuity")
            main = t.get("main_loop")
            cs = "n/a" if cont is None else f"{cont:.3f}"
            ms_ = "n/a" if main is None else f"{main:.3f}"
            L.append(f"| {g} | {mode.upper()} | completed | {cs} | {ms_} |")
    return "\n".join(L)


# --- report assembly --------------------------------------------------------

def _img(plots, key, alt):
    """Markdown image line for plots[key] if it rendered, else ''."""
    p = plots.get(key)
    return f"![{alt}]({os.path.basename(p)})\n\n" if p else ""


def build_report(runs, nsteps, title, prov, plots=None):
    plots = plots or {}
    L = [f"# {title}\n"]
    if prov:
        L.append(render_stamp(prov) + "\n")

    L.append("## Intent\n")
    L.append(
        "Quantify the cost of the continuity PPM sub-kernels ported to C++/AMReX, and "
        "split that cost into GPU **compute** versus the **data movement** the bridge "
        "performs around each call -- both the on-device repack kernels and the "
        "host<->device PCIe copies. The FMS mpp_clock timer reports only the sum; this "
        "report uses Nsight Systems to separate them.\n")
    L.append("<!-- commentary: key-finding -->\n")

    L.append("## Methodology\n")
    L.append(
        f"- One executable (`MOM6_using_TIM`, GPU offload + CUDA AMReX); the AMREX run "
        f"sets six `*_MODE=AMREX` env vars so the ported PPM kernels take the C++/AMReX "
        f"bridge. double_gyre, single rank on one A100, {nsteps} dynamic steps.\n"
        f"- GPU kernels attributed by demangled name: `MOM::` = ported PPM compute, "
        f"`turbotmp::copy_*` = bridge data repacking (host array <-> AMReX Array4 layout), other "
        f"`amrex::`/`turbotmp::` = AMReX infrastructure, rest = whole-model OpenMP "
        f"offload. PCIe copies split by API (`cudaMemcpyAsync` = bridge, "
        f"`cuMemcpy*Async_v2` = OpenMP).\n"
        f"- The mpp_clock continuity timer folds the AMReX call stack, the repacking "
        f"kernels, and the device copies together; only the Nsight split below "
        f"separates compute from data movement.\n"
        f"- NOTE: to run on the GPU at all this build needs the depth-list and restart "
        f"checksums disabled (`READ_DEPTH_LIST=False`, `RESTART_CONTROL=-1`); MOM6's "
        f"field_checksum routes to a TIM GPU reduction over a host pointer. See "
        f"PROFILING_DECISIONS.md.\n")
    L.append("<!-- commentary: methodology -->\n")

    L.append("## GPU kernel time by category (AMREX mode)\n")
    L.append(
        "From the AMREX-mode runs only. On-device GPU *kernel* time (the work the "
        "GPU executes); host<->device PCIe copies are not in these numbers -- they "
        "are reported separately below.\n")
    L.append(kernel_category_table(runs) + "\n")
    L.append("<!-- commentary: kernel-category -->\n")

    L.append("## Compute vs. data repacking -- the ported piece (AMREX mode)\n")
    L.append(
        "AMREX-mode runs only. Both bars in the figure are on-device GPU *kernel* "
        "time: PPM compute kernels vs. the data-repacking kernels (on-device layout "
        "conversion, Fortran array <-> AMReX Array4). The host<->device PCIe "
        "transfers are NOT kernel time and are not shown here -- they are reported "
        "separately in the PCIe section. The figure's bar labels give the "
        "data-repacking / compute time ratio.\n")
    L.append(_img(plots, "compute_vs_movement", "Compute vs data repacking"))
    L.append(compute_vs_repacking_table(runs) + "\n")
    L.append("<!-- commentary: compute-vs-repacking -->\n")

    L.append("## Continuity solver, end-to-end: AMReX vs OpenMP-Fortran (both modes)\n")
    L.append(
        "The full mpp_clock continuity wall time for each path, matched by problem "
        "size. This is the **folded** number: it INCLUDES on-device compute, the "
        "data-repacking kernels, AND the host<->device PCIe copies -- everything the "
        "solver does. (The Nsight breakdown above separates those pieces for the "
        "AMReX path.) In the figure, both axes are log-scaled and the per-point "
        "labels give the FORTRAN / AMREX wall-time ratio (>1 = AMReX faster).\n")
    L.append(_img(plots, "continuity_headtohead", "Continuity head-to-head"))
    L.append("<!-- commentary: continuity-headtohead -->\n")

    L.append("## Ported PPM compute kernels (AMREX mode)\n")
    L.append(
        "Per-kernel on-device GPU *compute* time (no data movement) for the ported "
        "PPM kernels, AMREX mode. The figure plots the kernels that launched; the "
        "table lists all six, with a Note on the three that launch no kernel of "
        "their own in this configuration.\n")
    L.append(_img(plots, "ported_kernels", "Ported PPM kernels"))
    L.append(ported_breakdown_table(runs) + "\n")
    L.append("<!-- commentary: ported-breakdown -->\n")

    L.append("## Bridge data-repacking kernels (AMREX mode)\n")
    L.append(
        "AMREX-mode runs only. On-device GPU time of the layout-conversion kernels "
        "(host Fortran array <-> AMReX Array4); the PCIe transfers these feed are "
        "separate, below.\n")
    L.append(repacking_table(runs) + "\n")
    L.append("<!-- commentary: repacking -->\n")

    L.append("## Host<->device PCIe copies (AMREX mode)\n")
    L.append(
        "From the AMREX-mode runs. The device-side H2D/D2H totals lump the bridge and "
        "the whole-model OpenMP offload together; the last two columns split memcpy "
        "*API* time by entry point (`cudaMemcpyAsync` = bridge, `cuMemcpy*` = OpenMP).\n")
    L.append(pcie_copy_table(runs) + "\n")
    L.append("<!-- commentary: pcie -->\n")

    L.append("## Folded mpp_clock timer (both modes)\n")
    L.append(
        "What the MOM timer alone reports (compute + repacking + copies, not "
        "separable from it), shown so the Nsight split can be read against it. Both "
        "FORTRAN and AMREX cells are listed (Mode column); FORTRAN cells also show "
        "run status.\n")
    L.append(folded_timer_table(runs) + "\n")
    L.append("<!-- commentary: folded-timer -->\n")

    if prov:
        L.append(render_provenance(prov, include_stamp=False) + "\n")
    return "\n".join(L)


# Matplotlib-capable pythons to try if the launching interpreter has none. The
# nvhpc python (needed for the right nsys) lacks matplotlib, so by default plotting
# would silently skip; instead we re-exec under one of these. Override with
# MATPLOTLIB_PYTHON, or pass --no-plots to opt out entirely.
_MPL_PYTHONS = [
    os.environ.get("MATPLOTLIB_PYTHON", ""),
    "/glade/u/apps/opt/conda/envs/npl/bin/python",
]


def _reexec_with_matplotlib_if_needed(args):
    """If plots are wanted but matplotlib is missing here, re-exec under a python
    that has it (keeping argv, incl. the inherited PATH so `nsys` still resolves).
    Guarded by an env sentinel so we re-exec at most once."""
    if "--no-plots" in sys.argv or os.environ.get("GEN_AMREX_REEXEC"):
        return
    try:
        import matplotlib  # noqa: F401
        return
    except ImportError:
        pass
    for py in _MPL_PYTHONS:
        if not py or not os.path.exists(py) or os.path.realpath(py) == os.path.realpath(sys.executable):
            continue
        # Confirm the candidate actually imports matplotlib before committing.
        chk = subprocess.run([py, "-c", "import matplotlib"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if chk.returncode == 0:
            sys.stderr.write(f"NOTE: re-exec under {py} for matplotlib (figures).\n")
            os.environ["GEN_AMREX_REEXEC"] = "1"
            os.execv(py, [py, os.path.abspath(__file__)] + sys.argv[1:])
    sys.stderr.write(
        "WARNING: no matplotlib-capable python found; rendering report WITHOUT "
        "figures. Set MATPLOTLIB_PYTHON or pass --no-plots.\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prof-dir", required=True)
    ap.add_argument("--stack-dir", help="path to turbo-stack checkout, for "
                    "recording commit hashes and build flags in the report")
    ap.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR,
                    help="parent directory for timestamped report dirs "
                    "(default: ../reports)")
    ap.add_argument("--label", default="amrex-continuity",
                    help="trailing label for the report dir name "
                    "(default: amrex-continuity); dir is <date-time>-<label>")
    ap.add_argument("--outdir", help="explicit output directory; overrides the "
                    "timestamped --reports-dir/--label naming")
    ap.add_argument("--nsteps", type=int, default=NSTEPS_DEFAULT)
    ap.add_argument("--title",
                    default="AMReX continuity port: compute vs. data movement")
    ap.add_argument("--note", default="",
                    help="free-form note added to the provenance block")
    ap.add_argument("--date", help="override the generated timestamp text in "
                    "the provenance block; defaults to now (YYYY-MM-DD HH:MM:SS)")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip figures (e.g. in an env without matplotlib)")
    args = ap.parse_args()

    if not args.no_plots:
        _reexec_with_matplotlib_if_needed(args)

    now = datetime.datetime.now()
    gen_time = args.date or now.isoformat(sep=" ", timespec="seconds")
    with tempfile.TemporaryDirectory() as workdir:
        runs = collect(args.prof_dir, workdir)
    if not runs:
        sys.exit(f"no prof_*.nsys-rep found in {args.prof_dir}")

    prov = gather_provenance(args.stack_dir, args.note, gen_time) \
        if args.stack_dir else None
    outdir = resolve_outdir(args, now)
    os.makedirs(outdir, exist_ok=True)

    plots = {}
    if not args.no_plots:
        try:
            plots["compute_vs_movement"] = plot_compute_vs_movement(
                runs, os.path.join(outdir, "compute_vs_movement.png"))
            plots["continuity_headtohead"] = plot_continuity_headtohead(
                runs, os.path.join(outdir, "continuity_headtohead.png"))
            plots["ported_kernels"] = plot_ported_kernels(
                runs, os.path.join(outdir, "ported_kernels.png"))
        except ImportError as e:
            sys.stderr.write(f"WARNING: plots skipped (matplotlib?): {e}\n"
                             f"  re-run with --no-plots to silence.\n")
            plots = {}

    report_path = os.path.join(outdir, "REPORT.md")
    with open(report_path, "w") as fh:
        fh.write(build_report(runs, args.nsteps, args.title, prov, plots))
    print(f"wrote {report_path}")
    for p in plots.values():
        if p:
            print(f"  {os.path.basename(p)}")
    print(f"  runs parsed: {', '.join(f'{m}_{i}' for (m, i) in sorted(runs))}")
    amrex_ok = [f"amrex_{i}" for i, _ in _amrex_runs(runs)]
    print(f"  AMREX cells with GPU data: {', '.join(amrex_ok) or 'NONE'}")


if __name__ == "__main__":
    main()
