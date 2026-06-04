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
  run-profile.sh         one-off Nsight Systems profiling run
  job-sweep-gpu.sh       PBS wrapper: submit the gpu sweep (1 GPU, ncpus=16)
  job-sweep-cpu.sh       PBS wrapper: submit the cpu sweep (1 node, ncpus=128)
  gen_report.py          parse run logs -> CSV + plots + Markdown report
docs/
  METHODOLOGY.md         scaling methodology (weak/strong, fixed-step trick)
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

To get the **"Continuity solver in isolation"** section (the GPU-ported region
on its own, separated from the whole-model main loop), the runs must expose
MOM6's `(Ocean continuity equation)` timer. That needs `clock_grain = 'ROUTINE'`
in the `&fms_nml` block of `input.nml` *at run time* — the FMS default of
`'NONE'` prints only the four top-level driver clocks. Runs made without it
still report fine; the continuity section just stays hidden. See
`docs/METHODOLOGY.md` for the timer caveat (the GPU continuity timer includes
device⇄host copies, not just the kernel).

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

## Relationship to turbo-stack

turbo-stack is a separate repository with its own submodules (MOM6, MARBL,
FMS2, TIM, …) and multi-gigabyte build artifacts. Keep it as a sibling
checkout (the default `TURBO_STACK` is `../turbo-stack-for-prof`) and reference
it via `$TURBO_STACK`; do not nest it inside this repo.
