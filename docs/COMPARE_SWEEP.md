# The four-config comparison sweep

This report type combines the problem-size **sweep** of the scaling report
(`docs/METHODOLOGY.md`) with a **multi-configuration comparison** originally
prototyped as a single-size harness in a scratch run directory
(`run.sh` + `extract_comparison.py`): every configuration runs the full
job-size sweep, each point repeated N times.

## The four configurations

| config | abbrev | build variant | hardware | PPM kernel routing |
|---|---|---|---|---|
| `dev_turbo_CPU` | dt_C | dev/turbo | min(i, 128) ranks | Fortran |
| `iturbo_CPU_amrex` | it_C_ax | iturbo | min(i, 128) ranks | AMReX |
| `dev_turbo_GPU` | dt_G | dev/turbo | 1 rank + 1 A100 | Fortran (OpenMP offload) |
| `iturbo_GPU_amrex` | it_G_ax | iturbo | 1 rank + 1 A100 | AMReX |

Abbreviations: dt=dev/turbo, it=iturbo; C=CPU, G=GPU; ax=AMReX kernels.

Each configuration is an executable resolved from a specific turbo-stack
checkout (`<stack>/bin/nvhpc/MOM6_using_TIM/MOM6/MOM6`; the stack roots are
overridable via `STACK_DEV_TURBO[_CPU]` / `STACK_ITURBO[_CPU]` env vars). The
`*_amrex` configs compare against their `dev_turbo` baseline on the same
hardware; the AMReX variant sets six environment variables that route the
ported continuity PPM kernels through C++/AMReX:

```
ZONAL_EDGE_THICKNESS_MODE / MERIDIONAL_EDGE_THICKNESS_MODE /
PPM_LIMIT_POS_MODE / PPM_LIMIT_CW84_MODE /
PPM_RECONSTRUCTION_X_MODE / PPM_RECONSTRUCTION_Y_MODE = AMREX
```

## Sweep methodology

Identical to the scaling sweep (`docs/METHODOLOGY.md`): job-size index `i`
sets a near-square layout of `i` 32x32 column blocks at NK=100, with
`dt = 1200/m`, and every run advances exactly **150 dynamic steps**
(`TIMEUNIT = dt` with `DAYMAX = 150`) so wall-clock is comparable across
sizes and configurations.

- **CPU configs** weak-scale: ranks = min(i, 128), constant 32x32x100
  gridpoints/rank up to the node cap, then per-rank work grows.
- **GPU configs** are a single-device problem-size scan: 1 rank, 1x1 layout.

On top of the sweep, each (config, size) point is repeated **N times**
(default 3) and the report averages timers across the repeats, reporting the
min-max main-loop spread.

## Timer sources

Two profiler worlds cover the five ported PPM kernels (`ppm_limit_pos`,
`PPM_reconstruction_x/y`, `zonal/meridional_edge_thickness`):

- **dev/turbo builds** time them with MOM6 `cpu_clock` — CLOCK_ROUTINE
  `mpp_clock` rows, exposed by `clock_grain = 'ROUTINE'` in `input.nml`.
- **iturbo AMReX builds** time them with the AMReX **TinyProfiler** INCLUSIVE
  table (BL_PROFILE ranges of the same names). All runs pass
  `tiny_profiler.device_synchronize_around_region=1` so kernel times are
  wall-clock comparable, and `tiny_profiler.print_threshold=0` so no row is
  dropped. These arguments are harmless to builds without TinyProfiler.

The report generator prefers TinyProfiler for iturbo configs and `cpu_clock`
for dev/turbo, falling back to the other; every kernel cell in the report is
annotated with its source. Both are inclusive wall-clock (launch + execution + synchronization),
so they are directly comparable. Caveats:

- Per-kernel timers are generally present **only in the GPU runs**; CPU kernel
  cells render as `n/a`.
- Aggregate clocks and `cpu_clock` kernel rows use the cross-PE **tavg** (mean)
  to stay consistent with the other report types. The original single-size
  harness used tmax (the slowest rank); for the single-rank GPU runs the two
  are identical.
- A GPU mpp_clock routine timer (e.g. `(Ocean continuity equation)`) folds in
  OpenMP `target ... map()` host<->device transfers and runtime overhead, not
  just kernels — see the caveat in `docs/METHODOLOGY.md`.

## Correctness cross-check

Each repeat's `ocean.stats` diagnostic file is kept as
`<config>_<i0>_run<r>.stats`. The report checks byte-identity across the
configs (and repeats) of each platform group per size — a cheap signal that
all variants computed the same physics. CPU and GPU groups are checked
separately, since their answers may legitimately differ in the last digits.

## Running it

```bash
qsub /path/to/turbo-prof/scripts/job-compare-cpu.sh
qsub /path/to/turbo-prof/scripts/job-compare-gpu.sh

# or a subset, e.g. a quick smoke test:
qsub -v CONFIGS="iturbo_GPU_amrex",JOBSIZES="1 4",NRUNS=1 \
    /path/to/turbo-prof/scripts/job-compare-gpu.sh
```

Both wrappers `cd` to the run directory (`RUN_DIR`, overridable via `qsub -v`)
— a double_gyre case directory with `MOM_input`, `diag_table`, and `input.nml`
with `clock_grain = 'ROUTINE'` — and call `run-compare-sweep.sh` once per
config. Logs land in the run directory as `<config>_<i0>_run<r>.out` (plus
`.err` and `.stats`).

## Generating the report

```bash
module load conda && conda activate npl
cd /path/to/turbo-prof/scripts
python3 gen_compare_report.py \
    --run-dir /glade/derecho/scratch/$USER/double_gyre_new \
    --label compare-sweep
```

Stack provenance defaults to the four standard `/glade/work` checkouts;
override with repeatable `--stack NAME=PATH` flags (names: `dev-turbo-cpu`,
`iturbo-cpu`, `dev-turbo`, `iturbo`). The report directory contains the
facts-only `report.md` (with `<!-- commentary: NAME -->` anchors for the
report-commentary skill), `results.csv` with one row per (config, size,
repeat), `provenance.json` covering all four stacks, and the plot PNGs.

## Nsight-profiled variant (leaf-kernel compute)

The sweep above sources its per-kernel times from stdout (mpp_clock /
TinyProfiler), which time each continuity call **end-to-end** — they fold the
GPU compute together with the host<->device PCIe copies the AMReX bridge (and
the OpenMP offload) perform around it. For a kernel-level view that isolates the
**compute** itself, the **two GPU configs** (`dev_turbo_GPU` vs
`iturbo_GPU_amrex`) are re-run under Nsight Systems by a **dedicated** script,
`run-nsys-compare-sweep.sh` (PBS wrapper `job-nsys-compare-gpu.sh`), and parsed
by `gen_nsys_compare_report.py`. The profiling sweep is a separate script --
exactly as `run-profile.sh` is separate from `run-scaling-sweep.sh` -- so the
plain comparison sweep's producer (`run-compare-sweep.sh`) is never touched and
keeps reproducing the previous report bit-for-bit. For the single-build AMReX
continuity story (compute vs. data movement, bridge repack, PCIe) see
`docs/AMREX_PROFILING.md`.

**Leaf kernels only, by design.** The report compares **only** the continuity
PPM *leaf* kernels — the bottom-of-tree compute kernels that appear as standalone
GPU kernels in *both* builds: `PPM_reconstruction_x`, `PPM_reconstruction_y`, and
the positive-definite / CW84 limiters (`ppm_limit_pos`, `ppm_limit_cw84`). These
are the only continuity kernels that pair apples-to-apples: the `edge_thickness`
wrappers are inlined in dev/turbo, and the rest of the solver (`mass_flux`,
`flux_adjust`, `set_*_bt_cont`, `convergence`) was never ported to AMReX, so it
has no `ParallelFor` counterpart. Excluded by construction: those wrappers, the
un-ported solver bulk, and the AMReX **bridge** (repack kernels + host<->device
copies). This is kernel **compute only** — `dev_turbo_GPU` = Fortran
`do concurrent`, `iturbo_GPU_amrex` = C++ AMReX `ParallelFor`.

**GPU-only, by construction.** Per-kernel device timers are GPU-only, so
`run-nsys-compare-sweep.sh` rejects CPU configs (use `run-compare-sweep.sh`
for those).

**Pairing across both builds.** nsys auto-demangles the kernel names, and the
`MOM::<routine>` reference survives in both the dev/turbo nvkernel names
(`mom_continuity_ppm_ppm_reconstruction_x_<line>_gpu`) and the iturbo AMReX names
(`...MOM::PPM_reconstruction_x(...)...`), so a substring match isolates each leaf
on both sides. A routine may emit several GPU loops; they are summed, and the
launch count is shown for both sides so any non-clean pairing (mismatched
launches) is visible.

**CSV-only at report time.** Every figure and table is derived from the per-trace
`prof_<config>_<i0>_run<r>_cuda_gpu_kern_sum.csv` files that
`run-nsys-compare-sweep.sh` dumps next to each trace (the kernel summary nsys
auto-demangles). So the report needs **only matplotlib** — no `nsys` and no
`c++filt` at report time. For mpp_clock-based continuity/throughput comparisons
(untraced, representative) use `gen_compare_report.py`. Traces are kept small with
`NSTEPS=40` (vs 150).

Running and generating:

```bash
# Profiled sweep: job-nsys-compare-gpu.sh runs the GPU configs under
# `nsys profile --stats`, writing prof_<config>_<i0>_run<r>.{nsys-rep,out,err}
# and dumping per-trace *_cuda_gpu_kern_sum.csv next to each (DUMP_CSV=0 to skip).
# Traced runs are slower and the .nsys-rep files are large; scope down with a
# shorter JOBSIZES for a quick check.
qsub /path/to/turbo-prof/scripts/job-nsys-compare-gpu.sh
# quick check, e.g.:
qsub -v JOBSIZES="4 16",NRUNS=1 \
    /path/to/turbo-prof/scripts/job-nsys-compare-gpu.sh

# Report (matplotlib only — reads the dumped CSVs, no nsys needed):
module load conda && conda activate npl
cd /path/to/turbo-prof/scripts
python3 gen_nsys_compare_report.py \
    --run-dir /glade/derecho/scratch/$USER/double_gyre_new \
    --label nsys-compare
```

The report directory holds `report.md` (anchors below); the plots `leaf_total.png`
(aggregate leaf compute on a log-log top panel + a linear do-concurrent/ParallelFor
ratio panel), `leaf_ratio.png` (per-leaf ratio), and `leaf_per_launch.png`
(per-launch time per leaf);
`results.csv` (one row per config/size/leaf with the summed GPU time `gpu_ms` and
`launches`); and a `provenance.json` covering the two GPU stacks. Commentary
anchors: `key-finding`, `methodology`, `total-compute`, `ratio-trend`,
`per-launch`, `leaf-comparison`.

All plots come from `cuda_gpu_kern_sum`, the only report `CSV_REPORTS` dumps.
