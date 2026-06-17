#!/usr/bin/env python3
"""Leaf-kernel comparison: Fortran `do concurrent` vs C++ AMReX `ParallelFor`.

A deliberately narrow report: it compares ONLY the continuity PPM **leaf** kernels
-- the bottom-of-tree compute kernels that appear as standalone GPU kernels in
*both* builds: `PPM_reconstruction_x/y` and the PPM positive-definite / CW84
limiters. These are the only continuity kernels that pair apples-to-apples:
wrappers (`edge_thickness`) are inlined in dev/turbo, and the rest of the solver
(`mass_flux`, `flux_adjust`, `set_*_bt_cont`, `convergence`) was never ported to
AMReX, so it has no `ParallelFor` counterpart.

Data source: the per-trace `prof_<config>_<i0>_run<r>_cuda_gpu_kern_sum.csv` files
that run-nsys-compare-sweep.sh dumps next to each trace. nsys's auto-demangled
names retain the `MOM::<routine>` reference, so a substring match isolates each
leaf. No nsys or c++filt needed at report time -- just the CSVs.

Pairing: `dev_turbo_GPU` = Fortran do-concurrent; `iturbo_GPU_amrex` = C++ AMReX
ParallelFor (AMREX mode). A routine may emit several GPU loops; they are summed,
and the launch count is shown for both sides so any non-clean pairing (mismatched
launches) is visible. This is **compute only** -- the AMReX bridge's repack
kernels and host<->device copies are NOT here (see the single-build AMReX report,
gen_amrex_report.py, for that). Facts-only; interpretation goes in the
`<!-- commentary: NAME -->` anchors.

Usage (matplotlib only; no nsys needed):
    python3 gen_nsys_compare_report.py --run-dir DIR [--stack NAME=PATH ...] \
        [--outdir DIR] [--label nsys-compare] [--no-plots]
"""

import argparse
import csv
import datetime
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turboprof.parsing import get_layout, NK, BLOCK
from turboprof.provenance import (
    gather_provenance_multi, render_provenance_multi, render_stamp)
from turboprof.reporting import DEFAULT_REPORTS_DIR, resolve_outdir
from gen_compare_report import DEFAULT_STACKS, REF_POINTS

# The two GPU configs being compared, as (config tag, provenance stack, label).
DC = ("dev_turbo_GPU",    "dev-turbo", "Fortran do concurrent")
PF = ("iturbo_GPU_amrex", "iturbo",    "C++ AMReX ParallelFor")

# Continuity PPM leaf kernels: (key, lowercase name substring, display label).
# The substring matches both the dev/turbo nvkernel names
# (`mom_continuity_ppm_ppm_reconstruction_x_<line>_gpu`) and the iturbo AMReX
# names (`...MOM::PPM_reconstruction_x(...)...`).
LEAVES = [
    ("PPM_reconstruction_x", "ppm_reconstruction_x", "PPM_reconstruction_x"),
    ("PPM_reconstruction_y", "ppm_reconstruction_y", "PPM_reconstruction_y"),
    ("ppm_limit_pos",        "ppm_limit_pos",        "PPM_limit_pos"),
    ("ppm_limit_cw84",       "ppm_limit_cw84",       "PPM_limit_CW84"),
]

_STYLE = {"PPM_reconstruction_x": dict(color="C0", marker="o"),
          "PPM_reconstruction_y": dict(color="C1", marker="s"),
          "ppm_limit_pos":        dict(color="C2", marker="^"),
          "ppm_limit_cw84":       dict(color="C3", marker="d")}


# --- collection (reads the dumped CSVs) -------------------------------------

def parse_leaf_csv(path):
    """Sum GPU time (ns) and launches per leaf from one cuda_gpu_kern_sum CSV."""
    out = {k: [0.0, 0] for k, _, _ in LEAVES}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Name") or "").lower()
            # A `_fortran` shim kernel (iturbo FORTRAN-mode, if ever present) is not
            # the AMReX ParallelFor leaf -- skip so it can't contaminate the match.
            if "fortran" in name:
                continue
            try:
                ns = float((row.get("Total Time (ns)") or "0").replace(",", ""))
                inst = int((row.get("Instances") or row.get("Count") or "0").replace(",", ""))
            except ValueError:
                continue
            for key, sub, _ in LEAVES:
                if sub in name:
                    out[key][0] += ns
                    out[key][1] += inst
                    break
    return out


def collect(run_dir, config):
    """Per size: averaged leaf {key:(ns, launches)} over repeats, from the dumped
    `prof_<config>_<i0>_run<r>_cuda_gpu_kern_sum.csv` files."""
    if not run_dir or not os.path.isdir(run_dir):
        return []
    pat = re.compile(rf"^prof_{re.escape(config)}_(\d+)_run(\d+)_cuda_gpu_kern_sum\.csv$")
    by_i = {}
    for fn in sorted(os.listdir(run_dir)):
        m = pat.match(fn)
        if not m:
            continue
        i = int(m.group(1))
        by_i.setdefault(i, []).append(parse_leaf_csv(os.path.join(run_dir, fn)))
    rows = []
    for i, reps in sorted(by_i.items()):
        gm, gn = get_layout(i)
        ni, nj = BLOCK * gm, BLOCK * gn
        leaves = {}
        for key, _, _ in LEAVES:
            present = [r[key] for r in reps if r[key][1] > 0]
            if present:
                leaves[key] = (sum(p[0] for p in present) / len(present),
                               sum(p[1] for p in present) / len(present))
        rows.append({"i": i, "ni": ni, "nj": nj, "gridpoints": ni * nj * NK,
                     "nruns": len(reps), "leaves": leaves})
    return rows


# --- plot -------------------------------------------------------------------

def plot_ratio(dc_rows, pf_rows, outpath):
    """do-concurrent / ParallelFor time vs problem size, one line per leaf."""
    import matplotlib.pyplot as plt
    dc = {r["i"]: r for r in dc_rows}
    pf = {r["i"]: r for r in pf_rows}
    fig, ax = plt.subplots(figsize=(8, 5))
    drawn = False
    for key, _, label in LEAVES:
        pts = []
        for i in sorted(set(dc) & set(pf)):
            d = dc[i]["leaves"].get(key)
            p = pf[i]["leaves"].get(key)
            if d and p and p[0] > 0:
                pts.append((dc[i]["gridpoints"], d[0] / p[0]))
        if not pts:
            continue
        drawn = True
        pts.sort()
        ax.plot([x for x, _ in pts], [y for _, y in pts], label=label, **_STYLE[key])
    if not drawn:
        plt.close(fig)
        return None
    ax.axhline(1.0, ls="-", color="black", alpha=0.6, lw=1)
    # Production operating-point reference(s), as in the other reports: dashed gray
    # vertical at each REF_POINTS gridpoint (e.g. 540x480x75 = 19.4M production).
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for gp, label in REF_POINTS:
        ax.axvline(gp, ls="--", alpha=0.6, color="gray")
        ax.text(gp, 0.02, " " + label, rotation=90, transform=trans,
                va="bottom", ha="right", fontsize=8, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("Problem size (total gridpoints = NI x NJ x 100)")
    ax.set_ylabel("do-concurrent / ParallelFor  (>1 = ParallelFor faster)")
    ax.set_title("Continuity leaf kernels: Fortran do-concurrent vs C++ ParallelFor")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


# Per-config line styles (color by programming model, not by leaf).
_CFG_STYLE = {"dc": dict(color="C0", marker="o", label="Fortran do concurrent"),
              "pf": dict(color="C3", marker="s", label="C++ AMReX ParallelFor")}
_XLABEL = "Problem size (total gridpoints = NI x NJ x 100)"


def _ref_lines(ax, label=False):
    """Dashed gray verticals at each production operating point (REF_POINTS)."""
    import matplotlib.transforms as mtransforms
    for gp, lab in REF_POINTS:
        ax.axvline(gp, ls="--", alpha=0.6, color="gray")
        if label:
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(gp, 0.02, " " + lab, rotation=90, transform=trans,
                    va="bottom", ha="right", fontsize=8, color="gray")


def plot_total(dc_rows, pf_rows, outpath):
    """Aggregate leaf compute (sum of all leaves) per config vs size, with the
    do-concurrent / ParallelFor ratio in a lower panel."""
    import matplotlib.pyplot as plt

    def totals(rows):
        out = {}
        for r in rows:
            s = sum(lv[0] for lv in r["leaves"].values())
            if s > 0:
                out[r["i"]] = (r["gridpoints"], s)
        return out

    dc, pf = totals(dc_rows), totals(pf_rows)
    if not dc and not pf:
        return None
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    for d, sty in ((dc, _CFG_STYLE["dc"]), (pf, _CFG_STYLE["pf"])):
        pts = sorted(d.values())
        if pts:
            ax.plot([gp for gp, _ in pts], [ns / 1e6 for _, ns in pts], **sty)
    rpts = [(dc[i][0], dc[i][1] / pf[i][1])
            for i in sorted(set(dc) & set(pf)) if pf[i][1] > 0]
    if rpts:
        rpts.sort()
        axr.plot([x for x, _ in rpts], [y for _, y in rpts], color="black", marker="d")
    axr.axhline(1.0, ls="-", color="gray", alpha=0.6, lw=1)
    _ref_lines(ax, label=True)
    _ref_lines(axr)
    ax.set_xscale("log")
    ax.set_yscale("log")   # compute spans ~3 decades; log-log shows all sizes (the
                           # ratio panel below carries the do-concurrent/PF gap)
    ax.set_ylabel("total leaf compute (ms / run)")
    ax.set_title("Aggregate continuity leaf compute: Fortran do-concurrent vs C++ ParallelFor")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    axr.set_ylabel("do-concurrent / ParallelFor")
    axr.set_xlabel(_XLABEL)
    axr.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


def plot_per_launch(dc_rows, pf_rows, outpath):
    """Per-launch average GPU time (total time / launches) per leaf, both configs
    vs problem size -- the per-kernel cost decoupled from launch count."""
    import matplotlib.pyplot as plt
    dc = {r["i"]: r for r in dc_rows}
    pf = {r["i"]: r for r in pf_rows}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    drawn = False
    for ax, (key, _, label) in zip(axes.flat, LEAVES):
        sub_drawn = False
        for rows, sty in ((dc, _CFG_STYLE["dc"]), (pf, _CFG_STYLE["pf"])):
            pts = []
            for i in sorted(rows):
                lv = rows[i]["leaves"].get(key)
                if lv and lv[1] > 0:
                    pts.append((rows[i]["gridpoints"], lv[0] / lv[1] / 1e3))  # us/launch
            if pts:
                pts.sort()
                ax.plot([x for x, _ in pts], [y for _, y in pts], **sty)
                drawn = sub_drawn = True
        ax.set_title(label, fontsize=10)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        _ref_lines(ax)
        if not sub_drawn:
            ax.text(0.5, 0.5, "no launches in these traces", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=9)
    if not drawn:
        plt.close(fig)
        return None
    axes.flat[0].legend(fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel(_XLABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel("us / launch")
    fig.suptitle("Per-launch kernel time per leaf: Fortran do-concurrent vs C++ ParallelFor")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return outpath


# --- tables -----------------------------------------------------------------

def grid_label(r):
    return f"{r['ni']}x{r['nj']}x{NK}"


def leaf_table(dc_rows, pf_rows, key):
    dc = {r["i"]: r for r in dc_rows}
    pf = {r["i"]: r for r in pf_rows}
    sizes = sorted(set(dc) | set(pf))
    body = []
    for i in sizes:
        d = dc.get(i, {}).get("leaves", {}).get(key)
        p = pf.get(i, {}).get("leaves", {}).get(key)
        if not d and not p:
            continue
        ref = dc.get(i) or pf.get(i)
        dn = int(round(d[1])) if d else None
        pn = int(round(p[1])) if p else None
        warn = " ⚠ launches differ" if (dn is not None and pn is not None and dn != pn) else ""
        ratio = f"{d[0] / p[0]:.3f}x" if (d and p and p[0]) else "-"
        body.append(
            f"| {grid_label(ref)} "
            f"| {d[0] / 1e6:,.2f} | {dn if dn is not None else '-'} "
            f"| {p[0] / 1e6:,.2f} | {pn if pn is not None else '-'} "
            f"| {ratio}{warn} |")
    if not body:
        return None
    head = ("| size | do-concurrent (ms) | launches | ParallelFor (ms) | launches "
            "| do-concurrent / ParallelFor |\n|---|--:|--:|--:|--:|--:|\n")
    return head + "\n".join(body)


def write_csv(dc_rows, pf_rows, path):
    cols = ["config", "impl", "i", "ni", "nj", "gridpoints", "nruns",
            "leaf", "gpu_ms", "launches"]
    out = []
    for tag, _, label, rows in ((DC[0], DC[1], DC[2], dc_rows),
                                (PF[0], PF[1], PF[2], pf_rows)):
        for r in rows:
            for key, _, _ in LEAVES:
                lv = r["leaves"].get(key)
                if not lv:
                    continue
                out.append({"config": tag, "impl": label, "i": r["i"],
                            "ni": r["ni"], "nj": r["nj"], "gridpoints": r["gridpoints"],
                            "nruns": r["nruns"], "leaf": key,
                            "gpu_ms": lv[0] / 1e6, "launches": lv[1]})
    if not out:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec in out:
            w.writerow(rec)


# --- report -----------------------------------------------------------------

def build_report(dc_rows, pf_rows, plots, stacks, title, prov=None):
    L = [f"# {title}\n"]
    if prov is not None:
        L.append(render_stamp(prov))

    L.append("## Intent\n")
    L.append(
        "The same continuity PPM **leaf** kernels, two ways on one A100: Fortran "
        "`do concurrent` (`dev_turbo_GPU`) vs C++ AMReX `ParallelFor` "
        "(`iturbo_GPU_amrex`). These are the only continuity kernels that pair "
        "apples-to-apples -- standalone in both builds, matching launch counts, "
        "identical computation. Crucially, both sides are timed by the **same** "
        "profiler (nsys), unlike the compare-sweep which mixes FMS mpp_clock with "
        "AMReX TinyProfiler -- so this is the single-clock check on the programming "
        "model's effect on kernel compute. The transitional AMReX bridge is "
        "excluded.\n")
    L.append(f"| | config | implementation | stack |\n|---|---|---|---|\n"
             f"| do-concurrent | `{DC[0]}` | {DC[2]} | `{stacks.get(DC[1], '?')}` |\n"
             f"| ParallelFor | `{PF[0]}` | {PF[2]} | `{stacks.get(PF[1], '?')}` |\n")
    L.append("\n<!-- commentary: key-finding -->\n")

    L.append("## Methodology\n")
    L.append(
        "- **Source:** per-trace `*_cuda_gpu_kern_sum.csv` from "
        "`run-nsys-compare-sweep.sh`; a substring match on the auto-demangled "
        "`MOM::<routine>` name isolates each leaf. No nsys at report time.\n"
        "- **Leaves:** `PPM_reconstruction_x/y`, `ppm_limit_pos`, `ppm_limit_cw84`; "
        "multiple GPU loops per routine are summed, and a size is flagged if the two "
        "sides' launch counts differ (not a clean pair).\n"
        "- **Excluded:** the inlined `edge_thickness` wrappers, the un-ported solver "
        "bulk, and the transitional AMReX bridge (repack + copies).\n"
        "- **Fair by construction:** same nvhpc 25.9 / `sm_80`, FMA off on both "
        "sides (`-Mnofma -Kieee` / `--fmad=false`), device code at `-O2` both ways -- "
        "so differences are intrinsic to the programming models, not the toolchain. "
        "Times are traced (Nsight) GPU kernel time.\n")
    L.append("<!-- commentary: methodology -->\n")

    if plots.get("total"):
        L.append("## Aggregate leaf compute\n")
        L.append(f"![Total leaf compute]({os.path.basename(plots['total'])})\n")
        L.append(
            "Sum of all leaf kernels' GPU time per run vs problem size; lower panel "
            "is the do-concurrent / ParallelFor ratio (>1 = ParallelFor faster). The "
            "headline compute number.\n")
        L.append("<!-- commentary: total-compute -->\n")

    if plots.get("ratio"):
        L.append("## Per-leaf ratio\n")
        L.append(f"![Leaf ratio]({os.path.basename(plots['ratio'])})\n")
        L.append(
            "Per-leaf do-concurrent / ParallelFor GPU-time ratio vs size "
            "(black line = parity; above it ParallelFor is faster).\n")
        L.append("<!-- commentary: ratio-trend -->\n")

    if plots.get("per_launch"):
        L.append("## Per-launch kernel time\n")
        L.append(f"![Per-launch time]({os.path.basename(plots['per_launch'])})\n")
        L.append(
            "GPU time for a single launch of each leaf (total / launches), "
            "decoupling per-kernel cost from launch count.\n")
        L.append("<!-- commentary: per-launch -->\n")

    L.append("## Per-leaf comparison\n")
    any_leaf = False
    for key, _, label in LEAVES:
        t = leaf_table(dc_rows, pf_rows, key)
        if not t:
            continue
        any_leaf = True
        L.append(f"### {label}\n")
        L.append(t)
        L.append("")
    if not any_leaf:
        L.append("_No leaf kernels found in the CSVs._\n")
    L.append("<!-- commentary: leaf-comparison -->\n")

    if prov is not None:
        L.append(render_provenance_multi(prov, include_stamp=False))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="directory with prof_<config>_<i>_run<r>_cuda_gpu_kern_sum.csv files")
    ap.add_argument("--stack", action="append", default=[], metavar="NAME=PATH",
                    help="turbo-stack checkout for provenance; repeatable "
                    f"(names: {DC[1]}, {PF[1]})")
    ap.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    ap.add_argument("--label", default="nsys-compare",
                    help="trailing label for the report dir (default: nsys-compare)")
    ap.add_argument("--outdir", help="explicit output dir; overrides timestamped naming")
    ap.add_argument("--title",
                    default="MOM6 continuity leaf kernels: do-concurrent vs ParallelFor")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--note", help="free-form note added to the provenance block")
    ap.add_argument("--date", help="override the generated timestamp text")
    args = ap.parse_args()

    stacks = {DC[1]: DEFAULT_STACKS[DC[1]], PF[1]: DEFAULT_STACKS[PF[1]]}
    for spec in args.stack:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--stack expects NAME=PATH, got: {spec}")
        stacks[name] = path

    now = datetime.datetime.now()
    gen_time = args.date or now.isoformat(sep=" ", timespec="seconds")

    dc_rows = collect(args.run_dir, DC[0])
    pf_rows = collect(args.run_dir, PF[0])
    print(f"  {DC[0]}: {len(dc_rows)} size(s); {PF[0]}: {len(pf_rows)} size(s)")

    prov = gather_provenance_multi(stacks, args.note, gen_time)
    outdir = resolve_outdir(args, now)
    os.makedirs(outdir, exist_ok=True)
    write_csv(dc_rows, pf_rows, os.path.join(outdir, "results.csv"))
    with open(os.path.join(outdir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    plots = {}
    if not args.no_plots:
        plots["total"] = plot_total(dc_rows, pf_rows, os.path.join(outdir, "leaf_total.png"))
        plots["ratio"] = plot_ratio(dc_rows, pf_rows, os.path.join(outdir, "leaf_ratio.png"))
        plots["per_launch"] = plot_per_launch(
            dc_rows, pf_rows, os.path.join(outdir, "leaf_per_launch.png"))

    report = build_report(dc_rows, pf_rows, plots, stacks, args.title, prov=prov)
    with open(os.path.join(outdir, "report.md"), "w") as fh:
        fh.write(report)

    print(f"Wrote report to {outdir}/")
    print("  report.md, results.csv, provenance.json")
    for p in plots.values():
        if p:
            print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
