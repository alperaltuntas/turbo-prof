# MOM6 scaling-test methodology (`run-scaling-sweep.sh`)

This script sweeps an idealized **double_gyre** MOM6 configuration across a range of
problem sizes to measure performance on CPU nodes and GPUs. This document explains how
a single loop variable drives the experiment and what each branch actually measures.

## The core idea

Everything keys off `JOBSIZES`, i.e. the loop variable `i`. One "unit" of `i` equals one
**32×32×100** block of grid cells (32×32 horizontal, `NK=100` vertical layers). From `i`:

- **Total problem size = 1024·`i` columns** (= 32m × 32n), growing *linearly* with `i`.
- `get_layout(i)` factors `i` into a near-square **m×n**, setting `NIGLOBAL=32m`,
  `NJGLOBAL=32n`.
- `dt = 1200/m`, `dt_therm = 2·dt`, and the key trick:
  **`TIMEUNIT = dt` with `DAYMAX = 150`** → every run is exactly **150 dynamic steps**,
  regardless of size. Wall-clock is therefore directly comparable across all runs, and the
  metric is *cost per step* (equivalently, columns·steps/sec throughput). Simulated days
  differ between runs, but that is irrelevant for a timing benchmark.

The `PLATFORM` branch only changes **how many ranks the domain is split across** — not the
domain itself.

## GPU branch (`NGPUS=1`) — single-device problem-size scan

Decomposition is always `get_layout(NGPUS)` = **1×1**, so `-np 1`. The whole growing domain
lands on **one GPU**:

| `i`  | domain m×n | NIGLOBAL×NJGLOBAL | MPI ranks (GPUs) | cols / GPU | rel. work |
|-----:|:----------:|:-----------------:|:----------------:|-----------:|----------:|
| 1    | 1×1        | 32×32             | 1                | 1,024      | 1×        |
| 2    | 2×1        | 64×32             | 1                | 2,048      | 2×        |
| 4    | 2×2        | 64×64             | 1                | 4,096      | 4×        |
| 8    | 4×2        | 128×64            | 1                | 8,192      | 8×        |
| 16   | 4×4        | 128×128           | 1                | 16,384     | 16×       |
| 32   | 8×4        | 256×128           | 1                | 32,768     | 32×       |
| 64   | 8×8        | 256×256           | 1                | 65,536     | 64×       |
| 128  | 16×8       | 512×256           | 1                | 131,072    | 128×      |
| 256  | 16×16      | 512×512           | 1                | 262,144    | 256×      |
| 512  | 32×16      | 1024×512          | 1                | 524,288    | 512×      |
| 1024 | 32×32      | 1024×1024         | 1                | 1,048,576  | 1024×     |

This is **neither weak nor strong scaling** — it is a single-device throughput/saturation
curve: how one GPU's cost-per-step grows with problem size, and where it saturates or runs
out of memory.

## CPU branch (`CPU_PER_NODE=128` on Derecho) — weak scaling, then capped

Ranks = `min(i, CPU_PER_NODE)` and the decomposition is `get_layout(nranks)`. Two regimes:

| `i`  | domain m×n | NI×NJ     | ranks | decomp lx×ly | cols / rank | regime                       |
|-----:|:----------:|:---------:|:-----:|:------------:|------------:|:-----------------------------|
| 1    | 1×1        | 32×32     | 1     | 1×1          | 1,024       | **weak scaling**             |
| 2    | 2×1        | 64×32     | 2     | 2×1          | 1,024       | weak                         |
| 4    | 2×2        | 64×64     | 4     | 2×2          | 1,024       | weak                         |
| 8    | 4×2        | 128×64    | 8     | 4×2          | 1,024       | weak                         |
| 16   | 4×4        | 128×128   | 16    | 4×4          | 1,024       | weak                         |
| 32   | 8×4        | 256×128   | 32    | 8×4          | 1,024       | weak                         |
| 64   | 8×8        | 256×256   | 64    | 8×8          | 1,024       | weak                         |
| 128  | 16×8       | 512×256   | 128   | 16×8         | 1,024       | weak (full node)             |
| 256  | 16×16      | 512×512   | 128   | 16×8         | 2,048       | **capped** (per-rank load 2×)|
| 512  | 32×16      | 1024×512  | 128   | 16×8         | 4,096       | capped (4×)                  |
| 1024 | 32×32      | 1024×1024 | 128   | 16×8         | 8,192       | capped (8×)                  |

- **`i` ≤ `CPU_PER_NODE`: textbook weak scaling.** Because `nranks = i` and the decomposition
  equals the domain factorization, every rank always owns exactly **32×32×100** cells. The
  problem grows in lockstep with rank count → flat per-step time means perfect scaling.
- **`i` > `CPU_PER_NODE`: full node, growing per-rank load.** Ranks pin at one node but the
  domain keeps growing, so each core does more work. This deliberately *mirrors the GPU
  branch* (fixed hardware, growing problem).

## Why the two branches are built this way

The overlap at `i ≥ CPU_PER_NODE` is the point: it puts **1 full CPU node** and **1 GPU** on
the *identical* problem sizes, so the **CPU-vs-GPU crossover** can be read straight off the
throughput curves — below some size the CPU node wins, above it the GPU wins (until it runs
out of memory).

So there are three studies in one script:

1. **CPU weak scaling** (`i` = 1…`CPU_PER_NODE`): does MOM6 hold flat as you fill a node?
2. **CPU node saturation** (`i` > `CPU_PER_NODE`): throughput of a saturated node vs. problem
   size.
3. **GPU single-device scan** (all `i`): throughput/memory limits of one GPU — directly
   comparable to (2).

The `JOBSIZES` comment variants (`# GPU-like` powers of 2 vs. `# CPU-like` with 96/192) just
pick grids that factor cleanly onto the respective node geometry (128 cores → 16×8; the
96/192 variants suit a 96-core node → 12×8).

## Routine-level breakdown (which offloaded routines are GPU-efficient)

The GPU build does **not** offload a single module — it GPU-ifies the model
*whole* via **OpenMP target directives** (`-mp=gpu`). The dynamical core,
tracers, and parameterizations are annotated with `!$omp target teams loop`
compute regions and explicit `map()` host⇄device data movement across ~28 source
files (`MOM_hor_visc`, `MOM_barotropic`, `MOM_CoriolisAdv`, `MOM_continuity_PPM`,
`MOM_vert_friction`, the pressure-force and tracer modules, …). So the
whole-model `Main loop` timer is *not* an Amdahl story about an un-ported
remainder; every routine runs on the GPU. The real question is **which offloaded
routines map efficiently to one GPU**, which needs the per-routine timers.

Those timers are *not* printed by default: the FMS `&fms_nml` namelist defaults
to `clock_grain = 'NONE'`, which emits only the four grain-0 driver clocks
(`Total runtime`, `Initialization`, `Main loop`, `Termination`). Set

```
&fms_nml
    clock_grain = 'ROUTINE'
/
```

in `input.nml` (the level MOM6's own `.testing` suite uses) so the full
routine-level table appears. `gen_report.py` then emits two sections:

- **"Where the main-loop time goes"** — a per-routine bar chart and table (the
  grain-31 timers) at one shared problem size, 1 CPU node vs 1 GPU. This is the
  crux view: continuity offloads well (faster on the GPU), while the barotropic
  solver and viscosity offload poorly (slower) and dominate the loop.
- **"Continuity solver in isolation"** — the continuity solver (the reviewer's
  focus and the largest single CPU routine) as a share of the main loop and its
  GPU-vs-CPU throughput ratio across problem sizes.

### Caveat: what a routine timer actually measures on the GPU

Each routine timer is an FMS `mpp_clock` around the routine **call**, so on the
GPU it folds in the OpenMP **`target ... map()` host⇄device transfers** and
runtime overhead around the offloaded loops, not just the kernel. So:

- It **is** the right number for *what the model pays for that routine
  end-to-end* (compute + transfers) — an honest integration cost.
- It **overstates** the GPU kernel and conflates compute with transfer, so it is
  **not** a measure of kernel speed.

To split kernel time from the host⇄device transfers, profile the region with
**Nsight Systems** (`run-profile.sh`). The barotropic solver — an iterative
sub-cycle with many small `target` regions, `map()` transfers, and halo updates
per step — is the leading suspect for the weak whole-model GPU number and the
natural next thing to profile.

## Fixed per-run settings

| Parameter         | Value         | Purpose                                              |
|-------------------|---------------|------------------------------------------------------|
| `NK`              | 100           | Vertical layers (constant)                           |
| `COORD_CONFIG`    | `linear`      | Linear coordinate over `DENSITY_RANGE`               |
| `DENSITY_RANGE`   | 2.0           | Stratification span                                  |
| `DAYMAX`          | 150           | In `TIMEUNIT=dt` → exactly 150 dynamic steps per run |
| `ENERGYSAVEDAYS`  | 50            | Energy diagnostic cadence                            |
| `DT` / `DT_THERM` | `1200/m` / 2× | Shrinks with domain to hold CFL as the grid grows    |
