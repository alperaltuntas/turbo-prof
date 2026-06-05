# Runbook: profiling the AMReX continuity port

End-to-end steps to produce the **AMReX continuity** report (the one
`gen_amrex_report.py` makes): profile the double_gyre case under Nsight Systems
for both the Fortran (OpenMP-offload) and the AMReX/CUDA continuity paths, across
a sweep of problem sizes, then build the Markdown report with figures.

For the *whole-model scaling* report (CPU vs GPU, `gen_report.py`) see
`REPORTING.md` / `METHODOLOGY.md` instead — this runbook is the AMReX-kernel flow.

## Prerequisites (once)

- A `MOM6_using_TIM` GPU executable built against **CUDA** AMReX:
  `turbo-stack-for-prof/bin/nvhpc/MOM6_using_TIM/MOM6/MOM6`
  (built by `build-cuda-amrex.sh` then `build-tim-gpu.sh` in turbo-stack; the MOM6
  submodule must be on the AMReX-port branch, e.g. `dev/turbo-debug`).
- A double_gyre run directory with `MOM_input`, `input.nml`, `diag_table`:
  `turbo-stack-for-prof/examples/double_gyre/` (the `RUNDIR` below).

```bash
RUNDIR=/glade/work/altuntas/turbo-stack-for-prof/examples/double_gyre
SCRIPTS=/glade/work/altuntas/turbo-prof/scripts
MODULES="ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3"
```

## The pieces

| Script | Role |
|---|---|
| `run-profile.sh [i] [MODE]` | one `nsys profile` run at size `i`, mode FORTRAN or AMREX. Owns the domain/timestep setup, the GPU-checksum overrides, and the nsys flags. Writes `prof_<mode>_<i>.{nsys-rep,out,err}` in the cwd. |
| `run-profile-sweep.sh "<sizes>" "<modes>"` | loops the above over sizes × modes (default: full `1..1024` × both modes). |
| `job-sweep-amrex.sh` | PBS wrapper (sibling of `job-sweep-cpu.sh`/`job-sweep-gpu.sh`): loads modules, `cd`s to the rundir, calls the sweep. |
| `gen_amrex_report.py` | parses the `prof_*` artifacts into `REPORT.md` + figures. |

Note: `run-profile.sh` disables two MOM6 checksums via `MOM_override`
(`READ_DEPTH_LIST=False`, `RESTART_CONTROL=-1`). On the TIM build MOM6's
`field_checksum` routes to a GPU reduction over a host pointer and crashes; these
overrides avoid both call sites. See `PROFILING_DECISIONS.md`.

## Step 1 — run the profiling sweep

Two ways. **Batch** is the normal path; **interactive** is for watching one run.

### Batch (recommended)

`job-sweep-amrex.sh` sweeps the full `1..1024` × {FORTRAN, AMREX} by default; to
scope it down, edit the `run-profile-sweep.sh` call at the bottom (it takes
optional `"<sizes>" "<modes>"` args). Then:

```bash
qsub $SCRIPTS/job-sweep-amrex.sh          # prints e.g. 6358250.desched1
```

Monitor it (state `Q` queued, `R` running, `F` finished):

```bash
qstat -u $USER                            # one-line view
qstat -xf 6358250.desched1 | grep job_state
```

The PBS `.o`/`.e` files land in the directory you submitted from; the per-size
MOM6 logs (`prof_<mode>_<i>.out`/`.err`) land in `RUNDIR`. Cancel with `qdel <JOBID>`.

### Interactive (to watch a single size)

Grab a GPU node, then drive `run-profile.sh` by hand:

```bash
qsub -I -A NCGD0067 -q main -l walltime=00:30:00 \
     -l select=1:ncpus=16:mpiprocs=1:ngpus=1
# once on the node:
module load $MODULES
cd $RUNDIR
sh $SCRIPTS/run-profile.sh 128 AMREX      # one size, one mode
sh $SCRIPTS/run-profile.sh 128 FORTRAN
# or the whole sweep:
sh $SCRIPTS/run-profile-sweep.sh "1 2 4 8 16 32 64 128 256 512 1024" "FORTRAN AMREX"
```

## Step 2 — sanity-check the runs

```bash
cd $RUNDIR
# Which cells completed? (a Main loop timer == it reached the end)
for f in prof_*.out; do
  printf "%-22s %s\n" "$f" "$(grep -m1 '^Main loop' "$f" || echo 'DID NOT COMPLETE')"
done
# Any aborts?
grep -l -iE "abort|CUDA error|partially present" prof_*.err
```

Expect `1024` to fail (single-A100 OOM). A `FATAL ... mismatched quote` means a
stray apostrophe got into `MOM_override` (keep comments out of it).

## Step 3 — build the report

Keep the **nvhpc modules loaded** (so `nsys` is the ≥2025.5 build that can read
the traces). You can launch with the plain python — the generator auto-re-execs
under a matplotlib-capable python (`npl` conda env) for the figures.

```bash
module load $MODULES
python3 $SCRIPTS/gen_amrex_report.py \
    --prof-dir $RUNDIR \
    --stack-dir /glade/work/altuntas/turbo-stack-for-prof
```

The output dir is auto-named `reports/<YYYY-MM-DD-HHMMSS>-amrex-continuity` (same
convention as `gen_report.py`); pass `--outdir DIR` to override or `--label NAME`
to change the trailing tag. Output: `REPORT.md` (facts only, with
`<!-- commentary: NAME -->` anchors) plus `compute_vs_movement.png`,
`continuity_headtohead.png`, `ported_kernels.png`.

Gotchas:
- **0 kernel rows / "no rows" warning** → wrong `nsys`. The cuda/12.9.0 nsys
  (2025.1) is older than the trace and exports an empty DB. Load nvhpc *after*
  cuda, or `export NSYS=/path/to/nvhpc/.../bin/nsys`.
- **No figures** → no matplotlib found anywhere. Set `MATPLOTLIB_PYTHON=/path/python`
  or pass `--no-plots`.

## Step 4 (optional) — add interpretation

In Claude Code, run the `report-commentary` skill pointed at the output dir to
fill the commentary anchors with grounded prose. See `REPORTING.md`.
