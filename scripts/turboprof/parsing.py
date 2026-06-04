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
}
_OVERRIDE_RE = {
    "niglobal": re.compile(r"'NIGLOBAL = (\d+)'"),
    "njglobal": re.compile(r"'NJGLOBAL = (\d+)'"),
    "dt":       re.compile(r"'DT = (\d+)'"),
}


def parse_run(path):
    """Return a dict of timers + parsed overrides, or None if no Main loop timer."""
    out = {"main_loop": None, "total_runtime": None, "init": None,
           "termination": None, "niglobal": None, "njglobal": None, "dt": None}
    with open(path, errors="replace") as fh:
        for line in fh:
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


def collect(run_dir, platform):
    """Parse all <platform>_<i>.out logs in run_dir into a list of metric dicts."""
    if not run_dir or not os.path.isdir(run_dir):
        return []
    pat = re.compile(rf"^{platform}_(\d+)\.out$")
    rows = []
    for fname in sorted(os.listdir(run_dir)):
        m = pat.match(fname)
        if not m:
            continue
        i = int(m.group(1))
        parsed = parse_run(os.path.join(run_dir, fname))
        if parsed is None:
            print(f"  warning: no Main loop timer in {fname} (run failed?)",
                  file=sys.stderr)
            continue

        gm, gn = get_layout(i)             # geometry layout (square-ish over i)
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
        })
    rows.sort(key=lambda r: r["i"])
    return rows


def add_throughput(rows, nsteps):
    for r in rows:
        r["nsteps"] = nsteps
        r["sec_per_step"] = r["main_loop"] / nsteps
        # cell-updates per second across the whole device/node
        r["throughput"] = r["gridpoints"] * nsteps / r["main_loop"]
    return rows
