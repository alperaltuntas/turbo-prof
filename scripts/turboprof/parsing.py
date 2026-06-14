"""Parse MOM6 double_gyre run logs into canonical per-run records.

The benchmarking sweep (run-scaling-sweep.sh) sweeps a job-size index ``i`` and
leaves one ``<platform>_<i>.out`` log per run. Each log ends with an FMS
mpp_clock table; we read the cross-PE mean ("tavg") of the named timers.

This module is report-type agnostic: it produces a list of dicts (one per run)
that any report generator can consume. ``add_throughput`` layers on the derived
metrics that depend on the (methodology-fixed) step count.
"""

import os
import re
import sys

# --- run geometry -----------------------------------------------------------

NK = 100               # vertical levels (#override NK = 100 in MOM_override)
BLOCK = 32             # one job-size unit is a 32x32 column block
CPU_PER_NODE = 128     # ranks are capped at one Derecho CPU node
NSTEPS_DEFAULT = 150   # dynamic steps per run, fixed by TIMEUNIT/DAYMAX trick


def get_layout(i):
    """Replicate the bash get_layout(): near-square m x n with m*n == i, m>=n."""
    m = 1
    while (m + 1) * (m + 1) <= i:
        m += 1
    while i % m != 0:
        m -= 1
    n = i // m
    if m < n:
        m, n = n, m
    return m, n


# --- log parsing ------------------------------------------------------------

# "Main loop   1   53.136911   53.138353   53.137857   0.000224 ..."
#  name        hits tmin       tmax        tavg        tstd
_TIMER_RE = {
    "main_loop":     re.compile(r"^Main loop\s+\d+\s+\S+\s+\S+\s+(\S+)"),
    "total_runtime": re.compile(r"^Total runtime\s+\d+\s+\S+\s+\S+\s+(\S+)"),
    "init":          re.compile(r"^Initialization\s+\d+\s+\S+\s+\S+\s+(\S+)"),
    "termination":   re.compile(r"^Termination\s+\d+\s+\S+\s+\S+\s+(\S+)"),
    # Continuity solver -- one of the OpenMP-GPU-offloaded routines and the
    # reviewer's focus. MOM6 wraps it in this CLOCK_MODULE timer, which appears
    # only when the run sets clock_grain >= 'MODULE' in input.nml (&fms_nml);
    # coarser runs leave it None. CAVEAT: this is an end-to-end timer around the
    # offloaded region. It folds in the `!$omp target ... map()` host<->device
    # transfers and runtime overhead, not just the GPU kernel; separating those
    # needs Nsight Systems (run-profile.sh).
    "continuity":    re.compile(r"^\(Ocean continuity equation\)\s+\d+\s+\S+\s+\S+\s+(\S+)"),
}
_OVERRIDE_RE = {
    "niglobal": re.compile(r"'NIGLOBAL = (\d+)'"),
    "njglobal": re.compile(r"'NJGLOBAL = (\d+)'"),
    "dt":       re.compile(r"'DT = (\d+)'"),
}


def _parse_timer_line(line):
    """Parse one FMS mpp_clock row into (name, {tavg, hits, tfrac, grain}).

    The clock name itself contains spaces (e.g. "(Ocean continuity equation)"),
    so parse from the right: the last nine columns are always
    ``hits tmin tmax tavg tstd tfrac grain pemin pemax``. Returns None for any
    line that is not a well-formed timer row.
    """
    toks = line.split()
    if len(toks) < 10:
        return None
    nums = toks[-9:]
    try:
        hits = int(nums[0])
        tavg = float(nums[3])
        tfrac = float(nums[5])
        grain = int(nums[6])
        int(nums[7]); int(nums[8])          # pemin/pemax -- presence check
    except ValueError:
        return None
    name = " ".join(toks[:-9])
    if not name:
        return None
    return name, {"tavg": tavg, "hits": hits, "tfrac": tfrac, "grain": grain}


def parse_run(path):
    """Return a dict of timers + parsed overrides, or None if no Main loop timer.

    ``timers`` holds the full mpp_clock table keyed by clock name (each value a
    {tavg, hits, tfrac, grain} dict), so report code can break the main loop
    down by component. The named scalar fields (main_loop, init, continuity, ...)
    are kept for convenience and backward compatibility.
    """
    out = {"main_loop": None, "total_runtime": None, "init": None,
           "termination": None, "continuity": None,
           "niglobal": None, "njglobal": None, "dt": None, "timers": {}}
    in_table = False
    with open(path, errors="replace") as fh:
        for line in fh:
            if "Tabulating mpp_clock" in line:
                in_table = True
            elif in_table:
                rec = _parse_timer_line(line)
                if rec and rec[0] not in out["timers"]:
                    out["timers"][rec[0]] = rec[1]
            for key, rgx in _TIMER_RE.items():
                if out[key] is None:
                    m = rgx.match(line)
                    if m:
                        out[key] = float(m.group(1))
            for key, rgx in _OVERRIDE_RE.items():
                if out[key] is None:
                    m = rgx.search(line)
                    if m:
                        out[key] = int(m.group(1))
    if out["main_loop"] is None:
        return None
    return out


_ERROR_HINT = re.compile(
    r"(out of memory|CUDA_ERROR|CUDA error|cuMemAlloc|Fatal Error|FATAL|"
    r"segmentation|killed|signal|insufficient|amrex::Abort|"
    r"illegal memory access|SIGABRT|SIGSEGV)", re.IGNORECASE)


def _error_reason(err_path):
    """Best-effort one-line failure cause from a run's stderr (.err) file.

    Returns the most informative error line (the model's CUDA/OOM/abort message
    is now captured there), or None. The GPU runtime prints thousands of
    "allocated block" lines before an OOM, so those are skipped. Among the
    matching lines the longest is kept: abort cascades end in terse lines
    ("SIGABRT") while the line naming the actual error is the verbose one.
    """
    if not os.path.isfile(err_path):
        return None
    hit = None
    with open(err_path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("allocated block") and _ERROR_HINT.search(line):
                if hit is None or len(line) > len(hit):
                    hit = line
    return hit[:200] if hit else None


def collect(run_dir, platform):
    """Parse all <platform>_<i>.out logs in run_dir. Returns (rows, failures).

    ``rows`` is one metric dict per run that reached the FMS timer table.
    ``failures`` lists runs whose log had no ``Main loop`` timer (crashed, ran
    out of memory, or never got that far), with the geometry reconstructed from
    the job-size index so callers can report what was lost instead of silently
    dropping it.
    """
    if not run_dir or not os.path.isdir(run_dir):
        return [], []
    pat = re.compile(rf"^{platform}_(\d+)\.out$")
    rows, failures = [], []
    for fname in sorted(os.listdir(run_dir)):
        m = pat.match(fname)
        if not m:
            continue
        i = int(m.group(1))
        gm, gn = get_layout(i)             # geometry layout (square-ish over i)
        parsed = parse_run(os.path.join(run_dir, fname))
        if parsed is None:
            print(f"  warning: no Main loop timer in {fname} (run failed?)",
                  file=sys.stderr)
            ni, nj = BLOCK * gm, BLOCK * gn
            failures.append({
                "i": i,
                "platform": platform,
                "ni": ni,
                "nj": nj,
                "nk": NK,
                "gridpoints": ni * nj * NK,
                "fname": fname,
                "reason": _error_reason(os.path.join(run_dir, fname[:-4] + ".err")),
            })
            continue

        ni = parsed["niglobal"] or BLOCK * gm
        nj = parsed["njglobal"] or BLOCK * gn
        nranks = min(i, CPU_PER_NODE) if platform == "cpu" else 1
        gridpoints = ni * nj * NK

        rows.append({
            "i": i,
            "platform": platform,
            "ni": ni,
            "nj": nj,
            "nk": NK,
            "nranks": nranks,
            "gridpoints": gridpoints,
            "gridpoints_per_rank": gridpoints / nranks,
            "dt": parsed["dt"],
            "main_loop": parsed["main_loop"],
            "total_runtime": parsed["total_runtime"],
            "init": parsed["init"],
            "termination": parsed["termination"],
            "continuity": parsed["continuity"],
            "timers": parsed["timers"],
        })
    rows.sort(key=lambda r: r["i"])
    failures.sort(key=lambda r: r["i"])
    return rows, failures


# --- comparison sweep (run-compare-sweep.sh) --------------------------------

# The five continuity PPM kernels instrumented in both profiler worlds:
# dev/turbo builds time them with MOM6 cpu_clock (CLOCK_ROUTINE mpp_clock rows),
# iturbo AMReX builds with the AMReX TinyProfiler (BL_PROFILE ranges of the same
# names). (display label, profiler name) pairs.
KERNELS = [
    ("ppm_limit_pos",             "ppm_limit_pos"),
    ("reconstruction_x",          "PPM_reconstruction_x"),
    ("reconstruction_y",          "PPM_reconstruction_y"),
    ("zonal_edge_thickness",      "zonal_edge_thickness"),
    ("meridional_edge_thickness", "meridional_edge_thickness"),
]

# A TinyProfiler row: name NCalls  v v v  pct%
_TINY_RE = re.compile(
    r"^(?P<name>[A-Za-z_]\w*)\s+(?P<ncalls>\d+)"
    r"\s+(?P<vmin>[-\d.eE+]+)\s+(?P<vavg>[-\d.eE+]+)\s+(?P<vmax>[-\d.eE+]+)\s+[\d.]+%\s*$")


def parse_tinyprofiler_incl(path):
    """Return {name: inclusive_avg_seconds} from the TinyProfiler INCLUSIVE table.

    AMReX prints two TinyProfiler tables at shutdown; the inclusive one is
    identified by its "Incl. ... NCalls" header. Returns {} if the run carries
    no TinyProfiler output (dev/turbo and iturbo-fortran builds).
    """
    out = {}
    in_incl = False
    with open(path, errors="replace") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if "Incl." in s and "NCalls" in s:
                in_incl = True              # header of the inclusive table
                continue
            if not in_incl:
                continue
            if set(s.strip()) <= {"-"} and s.strip():
                # a rule line ('-----'): the one right after the header, or the
                # one that closes the table. Close only after we've read rows.
                if out:
                    break
                continue
            m = _TINY_RE.match(s)
            if m:
                out[m.group("name")] = float(m.group("vavg"))
    return out


def average_dicts(dicts):
    """Given a list of {name: value} dicts, return {name: mean across the dicts}."""
    keys = set()
    for d in dicts:
        keys.update(d)
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _kernel_value(timers, tiny, prof_name, prefer_tiny):
    """One kernel's wall-clock (s) and its source ('tiny'|'mom6'|None).

    Uses the preferred profiler but falls back to the other, so iturbo-fortran
    builds (mpp_clock only) and AMReX builds (TinyProfiler) both report.
    A zero-hit mpp_clock row (kernel never launched) counts as absent.
    """
    rec = timers.get(prof_name)
    mom6 = rec["tavg"] if (rec and rec.get("hits", 0) > 0) else None
    order = (("tiny", tiny.get(prof_name)), ("mom6", mom6))
    if not prefer_tiny:
        order = order[::-1]
    for src, val in order:
        if val is not None:
            return val, src
    return None, None


def collect_compare(run_dir, config, platform, prefer_tiny):
    """Parse all <config>_<i0>_run<r>.out logs in run_dir. Returns (rows, failures).

    ``rows`` holds one dict per job size, shaped like ``collect()`` rows plus
    ``config``, ``nruns``, ``main_loop_min/max`` (repeat spread), ``kernels``
    (per-kernel seconds, see KERNELS), ``kernel_source``, and ``raw`` -- the
    unaveraged per-repeat records for results.csv. Scalar timers, the mpp_clock
    table, and kernel times are averaged across the repeats that completed.
    ``prefer_tiny`` picks the primary per-kernel profiler (TinyProfiler for
    iturbo configs, mpp_clock for dev/turbo); the other is the fallback.
    """
    if not run_dir or not os.path.isdir(run_dir):
        return [], []
    pat = re.compile(rf"^{re.escape(config)}_(\d+)_run(\d+)\.out$")
    by_i = {}
    failures = []
    for fname in sorted(os.listdir(run_dir)):
        m = pat.match(fname)
        if not m:
            continue
        i, run = int(m.group(1)), int(m.group(2))
        gm, gn = get_layout(i)
        parsed = parse_run(os.path.join(run_dir, fname))
        if parsed is None:
            print(f"  warning: no Main loop timer in {fname} (run failed?)",
                  file=sys.stderr)
            ni, nj = BLOCK * gm, BLOCK * gn
            failures.append({
                "i": i,
                "run": run,
                "config": config,
                "platform": platform,
                "ni": ni,
                "nj": nj,
                "nk": NK,
                "gridpoints": ni * nj * NK,
                "fname": fname,
                "reason": _error_reason(os.path.join(run_dir, fname[:-4] + ".err")),
            })
            continue
        parsed["run"] = run
        parsed["tiny"] = parse_tinyprofiler_incl(os.path.join(run_dir, fname))
        parsed["kernels"], parsed["kernel_source"] = {}, {}
        for disp, prof_name in KERNELS:
            parsed["kernels"][disp], parsed["kernel_source"][disp] = \
                _kernel_value(parsed["timers"], parsed["tiny"], prof_name,
                              prefer_tiny)
        by_i.setdefault(i, []).append(parsed)

    rows = []
    for i, reps in sorted(by_i.items()):
        reps.sort(key=lambda p: p["run"])
        gm, gn = get_layout(i)
        ni = reps[0]["niglobal"] or BLOCK * gm
        nj = reps[0]["njglobal"] or BLOCK * gn
        nranks = min(i, CPU_PER_NODE) if platform == "cpu" else 1
        gridpoints = ni * nj * NK

        # Average the mpp_clock table across repeats: tavg is the mean, the
        # hits/tfrac/grain metadata comes from the first repeat carrying it.
        timers = {}
        for p in reps:
            for name, rec in p["timers"].items():
                timers.setdefault(name, dict(rec))
        for name, tavgs in average_dicts(
                [{n: r["tavg"] for n, r in p["timers"].items()} for p in reps]).items():
            timers[name]["tavg"] = tavgs
        tiny = average_dicts([p["tiny"] for p in reps])

        kernels, kernel_source = {}, {}
        for disp, prof_name in KERNELS:
            kernels[disp], kernel_source[disp] = _kernel_value(
                timers, tiny, prof_name, prefer_tiny)

        main_loops = [p["main_loop"] for p in reps]
        rows.append({
            "i": i,
            "config": config,
            "platform": platform,
            "ni": ni,
            "nj": nj,
            "nk": NK,
            "nranks": nranks,
            "gridpoints": gridpoints,
            "gridpoints_per_rank": gridpoints / nranks,
            "dt": reps[0]["dt"],
            "nruns": len(reps),
            "main_loop": _mean(main_loops),
            "main_loop_min": min(main_loops),
            "main_loop_max": max(main_loops),
            "total_runtime": _mean([p["total_runtime"] for p in reps]),
            "init": _mean([p["init"] for p in reps]),
            "termination": _mean([p["termination"] for p in reps]),
            "continuity": _mean([p["continuity"] for p in reps]),
            "timers": timers,
            "kernels": kernels,
            "kernel_source": kernel_source,
            "raw": reps,
        })
    failures.sort(key=lambda r: (r["i"], r["run"]))
    return rows, failures


def add_throughput(rows, nsteps):
    for r in rows:
        r["nsteps"] = nsteps
        r["sec_per_step"] = r["main_loop"] / nsteps
        # cell-updates per second across the whole device/node
        r["throughput"] = r["gridpoints"] * nsteps / r["main_loop"]
        # Continuity-solver-only metrics, when the timer is present (runs with
        # clock_grain >= 'MODULE'). This isolates one offloaded routine from the
        # whole-model main loop -- see the CAVEAT on the continuity timer above:
        # the GPU figure here is compute + host<->device map transfers, not the
        # bare kernel.
        cont = r.get("continuity")
        if cont:
            r["continuity_per_step"] = cont / nsteps
            r["continuity_throughput"] = r["gridpoints"] * nsteps / cont
            r["continuity_frac"] = cont / r["main_loop"]
    return rows
