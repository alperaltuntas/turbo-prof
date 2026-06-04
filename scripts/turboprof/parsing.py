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
    r"(out of memory|CUDA_ERROR|cuMemAlloc|Fatal Error|FATAL|"
    r"segmentation|killed|signal|insufficient)", re.IGNORECASE)


def _error_reason(err_path):
    """Best-effort one-line failure cause from a run's stderr (.err) file.

    Returns the most informative error line (the model's CUDA/OOM/abort message
    is now captured there), or None. The GPU runtime prints thousands of
    "allocated block" lines before an OOM, so those are skipped.
    """
    if not os.path.isfile(err_path):
        return None
    hit = None
    with open(err_path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("allocated block") and _ERROR_HINT.search(line):
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
