# turbo-prof

Performance-analysis harness for the [turbo-stack](https://github.com/TURBO-ESM/turbo-stack)
ocean modeling stack (MOM6 + GPU port). This repository holds the **benchmarking
scripts, methodology, and generated reports** — it does *not* contain the
software stack itself. Point the scripts at a separate turbo-stack checkout.

## Layout

```
scripts/
  run-scaling-sweep.sh   scaling sweep; pass cpu or gpu (cpu = weak scaling
                         then saturated node; gpu = single-device problem scan)
  run-compare-sweep.sh   comparison sweep: one of four executable configs
                         (dev_turbo/iturbo_amrex x CPU/GPU)
                         across the job sizes, N repeats per point
  run-profile.sh         one-off Nsight Systems profiling run
  job-sweep-gpu.sh       PBS wrapper: submit the gpu sweep (1 GPU, ncpus=16)
  job-sweep-cpu.sh       PBS wrapper: submit the cpu sweep (1 node, ncpus=128)
  job-compare-gpu.sh     PBS wrapper: comparison sweep, the three GPU configs
  job-compare-cpu.sh     PBS wrapper: comparison sweep, the three CPU configs
  gen_report.py          parse run logs -> CSV + plots + facts-only Markdown report
  gen_compare_report.py  parse comparison-sweep logs -> four-config report
  strip_commentary.py    recover the facts-only report from an annotated one
docs/
  METHODOLOGY.md         scaling methodology (weak/strong, fixed-step trick)
  REPORTING.md           the facts-vs-commentary two-layer report design
  COMPARE_SWEEP.md       the four-config comparison sweep (configs, timer sources)
reports/
  <date-time>-<label>/   one committed report per run
```

## Running the sweeps

The PBS wrappers are self-contained — they hardcode the run directory and the
sweep-script path, so just submit them:

```bash
qsub /path/to/turbo-prof/scripts/job-sweep-gpu.sh   # or job-sweep-cpu.sh
```

Each wrapper `cd`s to the run directory (which holds `MOM_input`, `input.nml`,
`diag_table`) and runs `run-scaling-sweep.sh gpu|cpu`, leaving `<platform>_<i>.out`
logs there. `run-scaling-sweep.sh` self-locates the stack: it defaults `TURBO_STACK`
to a sibling `turbo-stack-for-prof` checkout next to this repo. Set `TURBO_STACK`
in the environment to point elsewhere.

## The comparison sweep

The comparison sweep runs **four executable configurations** — {dev/turbo,
iturbo with AMReX kernels} × {CPU, GPU} — across
the same job-size sweep, repeating each point N times (default 3) and parsing
both the FMS mpp_clock tables and (for AMReX runs) the AMReX TinyProfiler
table. See `docs/COMPARE_SWEEP.md` for the configs, timer sources, and caveats.

```bash
qsub /path/to/turbo-prof/scripts/job-compare-cpu.sh
qsub /path/to/turbo-prof/scripts/job-compare-gpu.sh
# subset / smoke test:
qsub -v CONFIGS="iturbo_GPU_amrex",JOBSIZES="1 4",NRUNS=1 \
    /path/to/turbo-prof/scripts/job-compare-gpu.sh
```

then, once both jobs finish:

```bash
python3 gen_compare_report.py --run-dir <run_dir> --label compare-sweep
```

## Generating a report

Run under an environment with matplotlib (e.g. the `npl` conda env on Derecho):

```bash
module load conda && conda activate npl
cd /path/to/turbo-prof/scripts
STACK=/path/to/turbo-stack-for-prof
python3 gen_report.py \
    --cpu-dir "$STACK/examples/double_gyre" \
    --gpu-dir "$STACK/examples/double_gyre" \
    --stack-dir "$STACK" \
    --label derecho-double_gyre \
    --title "MOM6 double_gyre GPU vs CPU scaling (Derecho)"
```

Both branches now share one run directory; the `cpu_*.out` and `gpu_*.out`
logs are namespaced by platform, so pass the same `double_gyre` dir for both
`--cpu-dir` and `--gpu-dir`.

To get the routine-level sections (**"Where the main-loop time goes"** and
**"Continuity solver in isolation"**), the runs must expose MOM6's per-routine
mpp_clock timers. That needs `clock_grain = 'ROUTINE'` in the `&fms_nml` block
of `input.nml` *at run time* — the FMS default of `'NONE'` prints only the four
top-level driver clocks. Runs made without it still report fine; those sections
just stay hidden. The GPU build offloads the model whole via OpenMP target
directives (`-mp=gpu`), so the breakdown shows which offloaded routines are
GPU-efficient. See `docs/METHODOLOGY.md` for the timer caveat (a GPU routine
timer includes the OpenMP `map()` host⇄device transfers, not just the kernel).

The report is written to a timestamped directory
`reports/<YYYY-MM-DD-HHMMSS>-<label>/`, so multiple reports per day stay
distinct and sort chronologically. Override the location with `--outdir DIR`
if you need an explicit path.

The `--stack-dir` flag records the turbo-stack commit, the MOM6 submodule
commit, the GPU build flags, and a dirty-tree warning in a **Provenance**
section, so every committed report is self-describing and reproducible. GPU is
optional — omit `--gpu-dir` for a CPU-only interim report and re-run later.

Each report directory contains `report.md`, `results.csv`, `provenance.json`,
and the plot PNGs. Raw per-run logs (`*_NNN.out`, `*.nsys-rep`, NetCDF) are not
committed; the CSV is the durable distillate.

## Adding commentary

`gen_report.py` emits a **facts-only** `report.md` — run parameters, plots, and
data tables, with no conclusions — that is reproducible from the logs. The
interpretation (bottleneck calls, what-it-means, what-to-profile-next) is a
separate layer added on top by the `report-commentary` Claude Code skill, which
fills the `<!-- commentary: NAME -->` anchors the script leaves behind, grounded
strictly in the report's own numbers. In Claude Code:

```
/report-commentary reports/<the-report-dir>
```

`strip_commentary.py` recovers the facts-only text from an annotated report, so
the two layers stay mechanically separable. See `docs/REPORTING.md` for the
design and `.claude/skills/report-commentary/` for the skill's grounding rules.

## Relationship to turbo-stack

turbo-stack is a separate repository with its own submodules (MOM6, MARBL,
FMS2, TIM, …) and multi-gigabyte build artifacts. Keep it as a sibling
checkout (the default `TURBO_STACK` is `../turbo-stack-for-prof`) and reference
it via `$TURBO_STACK`; do not nest it inside this repo.
