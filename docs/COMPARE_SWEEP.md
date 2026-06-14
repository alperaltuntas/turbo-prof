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
