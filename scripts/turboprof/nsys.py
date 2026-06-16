"""Nsight Systems (.nsys-rep) extraction for the GPU profiling report types.

`nsys profile --stats` (run-profile.sh, and run-compare-sweep.sh in PROFILE mode)
leaves one .nsys-rep trace per run. This module turns a trace into aggregated
kernel / memcpy / API-copy summaries via `nsys stats --report ... --format csv`,
classifying CUDA kernels into ported-PPM / bridge-repack / amrex / openmp buckets.

It is report-type agnostic: both the AMReX continuity report (gen_amrex_report.py)
and the nsys-compare report (gen_nsys_compare_report.py) consume
``summarize_rep()`` -- the former across Fortran-vs-AMReX MODE toggles of one build,
the latter across different builds (dev/turbo vs iturbo).

IMPORTANT: re-processing a .nsys-rep needs the SAME (or newer) nsys that recorded
it -- here the nvhpc-25.9 nsys (>=2025.5). The cuda/12.9.0 nsys (2025.1) is older
and exports an empty database. Load nvhpc after cuda, or set NSYS=/path/to/nsys.
"""

import csv
import os
import re
import subprocess
import sys

from .parsing import (
    parse_run, get_layout, average_dicts, _mean, _error_reason, NK, BLOCK)

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
# Two families, both verified against real traces: turbotmp:: C++ ParallelFor
# copies, and the Fortran `array_mod` do-concurrent copies (copy2{a,f}real{2,3}d).
REPACK_KERNELS = [
    ("copy_FortranHost_to_array4", "turbotmp: host -> Array4 pack"),
    ("copy_array4_to_FortranHost", "turbotmp: Array4 -> host unpack"),
    ("copy2areal",                 "array_mod: -> AMReX-real layout"),
    ("copy2freal",                 "array_mod: -> Fortran-real layout"),
]
# API names: which memcpy entry points belong to the AMReX bridge vs OpenMP offload.
BRIDGE_COPY_APIS = ("cudaMemcpyAsync", "cudaMemcpy")
OPENMP_COPY_APIS = ("cuMemcpy2DAsync_v2", "cuMemcpyDtoHAsync_v2",
                    "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoDAsync_v2")

# Fortran-offload continuity attribution. The dev/turbo (and iturbo-Fortran)
# continuity kernels are NOT C++ `MOM::` symbols -- nvhpc names each GPU kernel it
# generates from a `do concurrent` / `!$omp target` region after the enclosing
# Fortran routine and source line, e.g. `nvkernel_mom_continuity_ppm_<routine>_
# F1L<line>_<seq>`. The whole continuity solver lives in module
# `mom_continuity_ppm` (MOM_continuity_PPM.F90), so this module substring is a
# stable UMBRELLA match for "GPU compute attributable to the continuity solver",
# robust to inlining/fusion that would scramble per-routine names.
FORTRAN_CONTINUITY_MODULE = "continuity_ppm"
# Finer per-routine substrings, for a breakdown where the names survive (loops may
# fuse under inlining, in which case they fall under the umbrella with an
# "unresolved routine" key). Lowercased substring match; first match wins, so the
# leading entries are the routines actually observed as standalone kernels in the
# traces (mass_flux / flux_adjust / set_*_bt_cont / convergence); the trailing PPM
# leaves are the ported routines, which in dev/turbo are mostly inlined into the
# flux routines but may surface at other sizes.
FORTRAN_CONTINUITY_ROUTINES = [
    ("zonal_mass_flux",           "zonal mass flux"),
    ("meridional_mass_flux",      "meridional mass flux"),
    ("zonal_flux_adjust",         "zonal flux adjust"),
    ("meridional_flux_adjust",    "meridional flux adjust"),
    ("set_zonal_bt_cont",         "set zonal BT continuity"),
    ("set_merid_bt_cont",         "set meridional BT continuity"),
    ("zonal_convergence",         "zonal convergence"),
    ("meridional_convergence",    "meridional convergence"),
    ("ppm_reconstruction_x",      "PPM reconstruction x"),
    ("ppm_reconstruction_y",      "PPM reconstruction y"),
    ("ppm_limit_pos",             "PPM limiter: positive-definite"),
    ("ppm_limit_cw84",            "PPM limiter: CW84"),
    ("zonal_edge_thickness",      "zonal edge thickness"),
    ("meridional_edge_thickness", "meridional edge thickness"),
]

# The four `nsys stats` reports both collectors request from each trace.
KERN_REPORTS = ["cuda_gpu_kern_sum:mangled", "cuda_gpu_mem_time_sum",
                "cuda_gpu_mem_size_sum", "cuda_api_sum"]


# --- nsys CSV extraction ----------------------------------------------------

def _report_filebase(report):
    """nsys turns 'cuda_gpu_kern_sum:mangled' into the file suffix '..._mangled'."""
    return report.replace(":", "_")


def nsys_csv(rep_path, reports, workdir):
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


def _is_fortran_continuity(name_lower):
    """True if a demangled kernel name looks like a continuity-solver routine."""
    return (FORTRAN_CONTINUITY_MODULE in name_lower
            or any(k in name_lower for k, _ in FORTRAN_CONTINUITY_ROUTINES))


def summarize_kernels(kern_rows):
    """Aggregate cuda_gpu_kern_sum:mangled rows (names demangled) into buckets.

    The four ``buckets`` (ported / repack / amrex / openmp) partition all GPU
    kernel time and are unchanged. Three ADDITIONAL, additive views support the
    Fortran-offload configs, whose continuity kernels land in ``openmp`` (they are
    not C++ `MOM::` symbols): ``fortran_cont`` (total ns of `openmp` kernels whose
    name matches a continuity routine -- see FORTRAN_CONTINUITY_*), its per-routine
    breakdown ``per_fortran_cont``, and ``openmp_top`` (the largest *unmatched*
    `openmp` kernels, demangled, for auditing the name matcher against a real
    trace). ``fortran_cont`` is a labeled view INTO ``openmp``, not a 5th bucket,
    so the partition and every existing consumer stay intact.
    """
    names = [_col(r, "Name") for r in kern_rows]
    demangled = _demangle(names)
    buckets = {b: 0.0 for b in ("ported", "repack", "amrex", "openmp")}
    per_ported, per_repack, per_fortran_cont = {}, {}, {}
    fortran_cont = 0.0
    openmp_unmatched = []
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
        elif b == "openmp":
            if _is_fortran_continuity(name.lower()):
                fortran_cont += t
                k = _key_for(name, FORTRAN_CONTINUITY_ROUTINES) \
                    or "(continuity, unresolved routine)"
                d = per_fortran_cont.setdefault(k, {"ns": 0.0, "inst": 0.0})
                d["ns"] += t; d["inst"] += n
            else:
                openmp_unmatched.append((name, t))
    openmp_unmatched.sort(key=lambda x: -x[1])
    return {"buckets": buckets, "per_ported": per_ported, "per_repack": per_repack,
            "fortran_cont": fortran_cont, "per_fortran_cont": per_fortran_cont,
            "openmp_top": openmp_unmatched[:8]}


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


def summarize_rep(rep_path, out_path, workdir):
    """One .nsys-rep (+ its sibling .out) -> aggregated summary dict.

    Returns ``{kern, mem, api, timers, completed}``: the kernel/memcpy/API-copy
    summaries from the trace, plus the parsed FMS mpp_clock table from the stdout
    log (``timers``; None if the run never reached it) and a ``completed`` flag.
    Both GPU profiling reports build their per-run records on top of this.
    """
    csvs = nsys_csv(rep_path, KERN_REPORTS, workdir)
    timers = parse_run(out_path) if (out_path and os.path.exists(out_path)) else None
    return {
        "kern": summarize_kernels(csvs["cuda_gpu_kern_sum:mangled"]),
        "mem": summarize_memops(csvs["cuda_gpu_mem_time_sum"],
                                csvs["cuda_gpu_mem_size_sum"]),
        "api": summarize_api_copies(csvs["cuda_api_sum"]),
        "timers": timers,
        "completed": bool(timers),
    }


# --- nsys-compare sweep (run-nsys-compare-sweep.sh) -----------

def _avg_kern(kerns):
    """Average a list of summarize_kernels() dicts across repeats."""
    def avg_per(field):
        keys = set()
        for k in kerns:
            keys.update(k[field])
        out = {}
        for key in keys:
            ns = [k[field][key]["ns"] for k in kerns if key in k[field]]
            inst = [k[field][key]["inst"] for k in kerns if key in k[field]]
            if ns:
                out[key] = {"ns": sum(ns) / len(ns), "inst": sum(inst) / len(inst)}
        return out
    # openmp_top: merge the per-repeat unmatched lists by name (mean ns), keep top 8.
    merged = {}
    for k in kerns:
        for name, ns in k.get("openmp_top", []):
            merged.setdefault(name, []).append(ns)
    openmp_top = sorted(((nm, sum(v) / len(v)) for nm, v in merged.items()),
                        key=lambda x: -x[1])[:8]
    return {"buckets": average_dicts([k["buckets"] for k in kerns]),
            "per_ported": avg_per("per_ported"),
            "per_repack": avg_per("per_repack"),
            "fortran_cont": _mean([k["fortran_cont"] for k in kerns]) or 0.0,
            "per_fortran_cont": avg_per("per_fortran_cont"),
            "openmp_top": openmp_top}


def _avg_mem(mems):
    return {d: {"ns": _mean([m[d]["ns"] for m in mems]) or 0.0,
                "mb": _mean([m[d]["mb"] for m in mems]) or 0.0}
            for d in ("htod", "dtoh", "other")}


def _avg_api(apis):
    return {d: {"ns": _mean([a[d]["ns"] for a in apis]) or 0.0,
                "calls": _mean([a[d]["calls"] for a in apis]) or 0.0}
            for d in ("bridge", "openmp")}


def collect_nsys_compare(run_dir, config, platform, workdir):
    """Parse prof_<config>_<i0>_run<r>.{nsys-rep,out} logs. Returns (rows, failures).

    The Nsight analog of parsing.collect_compare: one row per job size, repeats
    averaged. Each row carries the same scalar/throughput shape as a compare-sweep
    row (so the shared throughput/continuity plots work) plus the nsys summaries
    ``kern`` / ``mem`` / ``api`` (averaged across repeats) and ``raw`` (per-repeat,
    for results.csv). Runs whose stdout never reached the FMS timer table go to
    ``failures`` with a best-effort reason from the .err file.

    ``platform`` is ``"gpu"`` for every profiled config (PROFILE mode is GPU-only);
    the parameter is kept for symmetry with collect_compare and the row schema.
    """
    if not run_dir or not os.path.isdir(run_dir):
        return [], []
    pat = re.compile(rf"^prof_{re.escape(config)}_(\d+)_run(\d+)\.nsys-rep$")
    by_i = {}
    failures = []
    for fname in sorted(os.listdir(run_dir)):
        m = pat.match(fname)
        if not m:
            continue
        i, run = int(m.group(1)), int(m.group(2))
        base = fname[:-len(".nsys-rep")]
        out_path = os.path.join(run_dir, base + ".out")
        summ = summarize_rep(os.path.join(run_dir, fname), out_path, workdir)
        if not summ["completed"]:
            print(f"  warning: no Main loop timer in {base}.out (run failed?)",
                  file=sys.stderr)
            gm, gn = get_layout(i)
            ni, nj = BLOCK * gm, BLOCK * gn
            failures.append({
                "i": i, "run": run, "config": config, "platform": platform,
                "ni": ni, "nj": nj, "nk": NK, "gridpoints": ni * nj * NK,
                "fname": fname,
                "reason": _error_reason(os.path.join(run_dir, base + ".err")),
            })
            continue
        summ["i"], summ["run"] = i, run
        by_i.setdefault(i, []).append(summ)

    rows = []
    for i, reps in sorted(by_i.items()):
        reps.sort(key=lambda s: s["run"])
        ts = [s["timers"] for s in reps]
        gm, gn = get_layout(i)
        ni = ts[0]["niglobal"] or BLOCK * gm
        nj = ts[0]["njglobal"] or BLOCK * gn
        gridpoints = ni * nj * NK

        # Average the mpp_clock table across repeats (same scheme as collect_compare).
        timers = {}
        for t in ts:
            for name, rec in t["timers"].items():
                timers.setdefault(name, dict(rec))
        for name, tavg in average_dicts(
                [{n: r["tavg"] for n, r in t["timers"].items()} for t in ts]).items():
            timers[name]["tavg"] = tavg

        main_loops = [t["main_loop"] for t in ts]
        rows.append({
            "i": i, "config": config, "platform": platform,
            "ni": ni, "nj": nj, "nk": NK, "nranks": 1,
            "gridpoints": gridpoints, "gridpoints_per_rank": gridpoints,
            "dt": ts[0]["dt"],
            "nruns": len(reps),
            "main_loop": _mean(main_loops),
            "main_loop_min": min(main_loops), "main_loop_max": max(main_loops),
            "total_runtime": _mean([t["total_runtime"] for t in ts]),
            "init": _mean([t["init"] for t in ts]),
            "termination": _mean([t["termination"] for t in ts]),
            "continuity": _mean([t["continuity"] for t in ts]),
            "timers": timers,
            "kern": _avg_kern([s["kern"] for s in reps]),
            "mem": _avg_mem([s["mem"] for s in reps]),
            "api": _avg_api([s["api"] for s in reps]),
            "raw": reps,
        })
    failures.sort(key=lambda r: (r["i"], r["run"]))
    return rows, failures
