# MOM6 double_gyre four-config comparison sweep

**Generated:** 2026-06-13 22:51:16 on `derecho6`

## Intent

Compare four MOM6 `double_gyre` configurations -- {dev/turbo, iturbo-AMReX} x {CPU, GPU} -- across the standard problem-size sweep on Derecho. Each pairs an executable from a specific turbo-stack checkout with, for the AMReX variants, the six `*_MODE=AMREX` env vars that route the ported continuity PPM kernels through C++/AMReX.

| config | abbrev | stack | resources | PPM kernel routing |
|---|---|---|---|---|
| `dev_turbo_CPU` | dt_C | `dev-turbo-cpu` (`/glade/work/altuntas/turbo-stack-dev-turbo-cpu`) | min(i, 128) ranks | Fortran |
| `iturbo_CPU_amrex` | it_C_ax | `iturbo-cpu` (`/glade/work/altuntas/turbo-stack-iturbo-cpu`) | min(i, 128) ranks | AMReX (`*_MODE=AMREX`) |
| `dev_turbo_GPU` | dt_G | `dev-turbo` (`/glade/work/altuntas/turbo-stack-dev-turbo`) | 1 rank + 1 GPU | Fortran |
| `iturbo_GPU_amrex` | it_G_ax | `iturbo` (`/glade/work/altuntas/turbo-stack-iturbo`) | 1 rank + 1 GPU | AMReX (`*_MODE=AMREX`) |


<!-- commentary: key-finding -->

## Methodology

Each run advances exactly **150 dynamic steps** (`TIMEUNIT = dt`, `DAYMAX = 150`), so wall-clock is comparable across sizes and configs. Job-size index `i` is a near-square layout of `i` 32x32 blocks at NK=100.

- **CPU configs**: weak scaling -- ranks grow with `i` at a constant 32x32x100 gridpoints/rank up to the 128-rank node cap, then stay at 128 while per-rank work grows.
- **GPU configs**: single-device scan (1 rank, 1 A100, 1x1).

Each (config, size) point runs **N times** (`runs` columns); timers are averaged, and `spread` is the min-max of the main-loop timer over repeats.

Aggregate timers are the cross-PE mean ("tavg") of FMS `mpp_clock` rows (e.g. `Main loop`, `(Ocean continuity equation)`). Per-kernel timers come from two sources -- dev/turbo: MOM6 `cpu_clock`; iturbo-AMReX: AMReX TinyProfiler INCLUSIVE (run with `tiny_profiler.device_synchronize_around_region=1`) -- each falling back to the other, with every kernel cell annotated by source. Both are inclusive wall-clock (launch + execution + sync), so directly comparable. (This report uses mpp_clock tavg; the single-size harness used tmax -- identical for single-rank GPU runs.) See `docs/COMPARE_SWEEP.md`.

<!-- commentary: methodology -->

## Throughput vs problem size (whole model)

![Throughput](throughput.png)

Cell-updates/s vs problem size, all four configs. **Color** = platform (CPU blue, GPU orange); **marker/linestyle** = build variant (dev/turbo solid squares, iturbo-AMReX dashed circles). Dotted red vertical: i=128 (13.1M gridpoints), where the CPU run fills a full Derecho CPU node (128 cores) -- a clean 128-cores-vs-1-GPU comparison. Dashed gray: the 19.4M production point. This is a **whole-model** rate (total cell-updates / main-loop time), so for iturbo it includes the per-call host<->device bridge marshalling, not just the kernels -- see the kernel-only throughput below.

<!-- commentary: throughput -->

## Throughput vs problem size: compute kernels only

![Kernel throughput](kernel_throughput.png)

The same cell-updates/s metric from the continuity kernel **compute** time alone (outer zonal + meridional `edge_thickness`), excluding the AMReX bridge's host<->device marshalling; for iturbo, its gap from the whole-model throughput above is the bridge overhead. Encoding and verticals as above.

"Compute" here means kernel **launch + on-device execution + sync**, not pure arithmetic -- both configs are timed on this same basis (`device_synchronize_around_region=1`; the dev/turbo `do concurrent` launches are host-synchronous), and only the bridge transfers are excluded, so the comparison is fair.

<!-- commentary: kernel-throughput -->

## iturbo vs dev/turbo speedup: kernels only

![Kernel speedup](kernel_speedup.png)

The same ratio restricted to the **continuity kernel compute** (outer `zonal_edge_thickness` + `meridional_edge_thickness`; `BL_PROFILE` for iturbo, `cpu_clock` for dev/turbo), excluding the bridge's H2D/D2H copies -- isolating the port's compute from its integration overhead. Its gap from the whole-model throughput/continuity curves is the bridge tax. CPU and GPU pairs are shown where timers exist; encoding and verticals as above.

<!-- commentary: kernel-speedup -->

## Head-to-head: CPU configs

Main-loop seconds for the three CPU configurations at each size, with each iturbo variant's speedup vs dev/turbo (>1 = iturbo faster). Missing cells are sizes that config did not complete.

| i | gridpoints | dt_C (s) | it_C_ax (s) | it_C_ax speedup |
|---|---|---|---|---|
| 1 | 102,400 | 18.447 | 25.808 | 0.71x |
| 2 | 204,800 | 22.088 | 29.578 | 0.75x |
| 4 | 409,600 | 24.593 | 32.676 | 0.75x |
| 8 | 819,200 | 32.326 | 43.021 | 0.75x |
| 16 | 1,638,400 | 50.606 | 69.498 | 0.73x |
| 32 | 3,276,800 | 52.204 | 70.789 | 0.74x |
| 64 | 6,553,600 | 55.733 | 74.382 | 0.75x |
| 128 | 13,107,200 | 58.645 | 76.633 | 0.77x |
| 256 | 26,214,400 | 118.7 | 154.5 | 0.77x |
| 512 | 52,428,800 | 309.5 | 411.8 | 0.75x |
| 1024 | 104,857,600 | 613.5 | 812.2 | 0.76x |


## Head-to-head: GPU configs

Main-loop seconds for the three GPU configurations at each size, with each iturbo variant's speedup vs dev/turbo (>1 = iturbo faster). Missing cells are sizes that config did not complete.

| i | gridpoints | dt_G (s) | it_G_ax (s) | it_G_ax speedup |
|---|---|---|---|---|
| 1 | 102,400 | 17.854 | 18.963 | 0.94x |
| 2 | 204,800 | 18.077 | 20.244 | 0.89x |
| 4 | 409,600 | 19.890 | 23.208 | 0.86x |
| 8 | 819,200 | 21.809 | 26.586 | 0.82x |
| 16 | 1,638,400 | 26.358 | 35.888 | 0.73x |
| 32 | 3,276,800 | 35.372 | 54.032 | 0.65x |
| 64 | 6,553,600 | 54.349 | 91.856 | 0.59x |
| 128 | 13,107,200 | 92.963 | 166.4 | 0.56x |
| 256 | 26,214,400 | 164.6 | 325.6 | 0.51x |
| 512 | 52,428,800 | 395.4 | 875.1 | 0.45x |


<!-- commentary: head-to-head -->

## Continuity solver

![Continuity](continuity.png)

The `(Ocean continuity equation)` mpp_clock timer vs problem size, all configs -- the routine whose PPM kernels the AMReX port replaces, timed end-to-end. For iturbo this folds in the host<->device transfers and runtime overhead, not just kernels (hence "includes bridge"). Verticals as above.

<!-- commentary: continuity -->

## Ported PPM kernels

![Kernels](kernels.png)

Wall-clock of the five ported PPM kernels vs problem size, all configs (timer sources per Methodology; launch + execution + sync, bridge transfers excluded). The kernels nest (`edge_thickness` > `reconstruction` > `limiter`), so rows are inclusive, not additive.

<!-- commentary: kernels -->

## Kernel snapshot at i=512

All configurations side by side at job size i=512 (the largest size completed by every config). Seconds, averaged over repeats; each kernel cell notes its timer source (`mom6` cpu_clock or `tiny` TinyProfiler inclusive).

| timer | dt_C | it_C_ax | dt_G | it_G_ax |
|---|---|---|---|---|
| ppm_limit_pos | 6.562 (mom6) | 7.231 (tiny) | 1.372 (mom6) | 1.315 (tiny) |
| reconstruction_x | 14.634 (mom6) | 13.880 (tiny) | 2.153 (mom6) | 2.042 (tiny) |
| reconstruction_y | 12.856 (mom6) | 13.087 (tiny) | 2.196 (mom6) | 2.102 (tiny) |
| zonal_edge_thickness | 14.678 (mom6) | 13.883 (tiny) | 2.167 (mom6) | 2.045 (tiny) |
| meridional_edge_thickness | 12.876 (mom6) | 13.090 (tiny) | 2.211 (mom6) | 2.105 (tiny) |
| continuity (mpp_clock) | 119.7 | 219.1 | 47.703 | 529.9 |
| main loop (mpp_clock) | 309.5 | 411.8 | 395.4 | 875.1 |


Speedup vs the dev/turbo baseline on the same hardware (>1 = iturbo variant faster):

| timer | it_C_ax | it_G_ax |
|---|---|---|
| ppm_limit_pos | 0.91x | 1.04x |
| reconstruction_x | 1.05x | 1.05x |
| reconstruction_y | 0.98x | 1.04x |
| zonal_edge_thickness | 1.06x | 1.06x |
| meridional_edge_thickness | 0.98x | 1.05x |
| continuity | 0.55x | 0.09x |
| main loop | 0.75x | 0.45x |


<!-- commentary: kernel-snapshot -->

## ocean.stats cross-check

Byte-identity of the `ocean.stats` diagnostic file across the configs (and repeats) of each platform group, per size -- a cheap signal that all variants computed the same physics. CPU and GPU groups are checked separately.

| i | CPU configs | GPU configs |
|---|---|---|
| 1 | identical (2 configs) | identical (2 configs) |
| 2 | identical (2 configs) | identical (2 configs) |
| 4 | identical (2 configs) | identical (2 configs) |
| 8 | identical (2 configs) | identical (2 configs) |
| 16 | identical (2 configs) | identical (2 configs) |
| 32 | identical (2 configs) | identical (2 configs) |
| 64 | identical (2 configs) | identical (2 configs) |
| 128 | identical (2 configs) | identical (2 configs) |
| 256 | identical (2 configs) | identical (2 configs) |
| 512 | identical (2 configs) | identical (2 configs) |
| 1024 | identical (2 configs) | identical (1 configs) |


<!-- commentary: ocean-stats -->

## Failed / missing runs

These runs produced no FMS `Main loop` timer, so they did not complete and are excluded from the plots and tables above (their repeats that did complete are still averaged). The `cause` column is the failing line from the run's stderr.

| config | i | run | NI x NJ | gridpoints | log | cause (from stderr) |
|---|---|---|---|---|---|---|
| `dev_turbo_GPU` | 1024 | 1 | 1024x1024 | 104,857,600 | `dev_turbo_GPU_1024_run1.out` | Accelerator Fatal Error: call to cuMemAlloc returned error 2 (CUDA_ERROR_OUT_OF_MEMORY): Out of memory |
| `dev_turbo_GPU` | 1024 | 2 | 1024x1024 | 104,857,600 | `dev_turbo_GPU_1024_run2.out` | Accelerator Fatal Error: call to cuMemAlloc returned error 2 (CUDA_ERROR_OUT_OF_MEMORY): Out of memory |
| `dev_turbo_GPU` | 1024 | 3 | 1024x1024 | 104,857,600 | `dev_turbo_GPU_1024_run3.out` | Accelerator Fatal Error: call to cuMemAlloc returned error 2 (CUDA_ERROR_OUT_OF_MEMORY): Out of memory |
| `iturbo_GPU_amrex` | 512 | 3 | 1024x512 | 52,428,800 | `iturbo_GPU_amrex_512_run3.out` | deg0037.hsn.de.hpc.ucar.edu: rank 0 died from signal 15 |
| `iturbo_GPU_amrex` | 1024 | 1 | 1024x1024 | 104,857,600 | `iturbo_GPU_amrex_1024_run1.out` | Accelerator Fatal Error: call to cuMemAlloc returned error 2 (CUDA_ERROR_OUT_OF_MEMORY): Out of memory |
| `iturbo_GPU_amrex` | 1024 | 2 | 1024x1024 | 104,857,600 | `iturbo_GPU_amrex_1024_run2.out` | Accelerator Fatal Error: call to cuMemAlloc returned error 2 (CUDA_ERROR_OUT_OF_MEMORY): Out of memory |
| `iturbo_GPU_amrex` | 1024 | 3 | 1024x1024 | 104,857,600 | `iturbo_GPU_amrex_1024_run3.out` | Accelerator Fatal Error: call to cuMemAlloc returned error 2 (CUDA_ERROR_OUT_OF_MEMORY): Out of memory |


<!-- commentary: failures -->

## Results by configuration

### dev_turbo_CPU (dt_C)

| i | ranks | NI x NJ | gridpoints | dt | runs | main loop (s) | spread (s) | s/step | throughput (cell-up/s) | continuity (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 32x32 | 102,400 | 1200 | 3 | 18.447 | 18.308-18.565 | 0.1230 | 8.326e+05 | 9.693 |
| 2 | 2 | 64x32 | 204,800 | 600 | 3 | 22.088 | 22.007-22.214 | 0.1473 | 1.391e+06 | 10.788 |
| 4 | 4 | 64x64 | 409,600 | 600 | 3 | 24.593 | 24.454-24.852 | 0.1640 | 2.498e+06 | 11.601 |
| 8 | 8 | 128x64 | 819,200 | 300 | 3 | 32.326 | 31.952-33.024 | 0.2155 | 3.801e+06 | 14.035 |
| 16 | 16 | 128x128 | 1,638,400 | 300 | 3 | 50.606 | 50.548-50.705 | 0.3374 | 4.856e+06 | 20.983 |
| 32 | 32 | 256x128 | 3,276,800 | 150 | 3 | 52.204 | 52.137-52.237 | 0.3480 | 9.415e+06 | 20.863 |
| 64 | 64 | 256x256 | 6,553,600 | 150 | 3 | 55.733 | 55.574-55.850 | 0.3716 | 1.764e+07 | 21.425 |
| 128 | 128 | 512x256 | 13,107,200 | 75 | 3 | 58.645 | 58.507-58.770 | 0.3910 | 3.352e+07 | 21.133 |
| 256 | 128 | 512x512 | 26,214,400 | 75 | 3 | 118.655 | 118.214-119.202 | 0.7910 | 3.314e+07 | 43.161 |
| 512 | 128 | 1024x512 | 52,428,800 | 37 | 3 | 309.487 | 308.792-310.052 | 2.0632 | 2.541e+07 | 119.7 |
| 1024 | 128 | 1024x1024 | 104,857,600 | 37 | 3 | 613.529 | 612.903-614.767 | 4.0902 | 2.564e+07 | 236.0 |

### iturbo_CPU_amrex (it_C_ax)

| i | ranks | NI x NJ | gridpoints | dt | runs | main loop (s) | spread (s) | s/step | throughput (cell-up/s) | continuity (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 32x32 | 102,400 | 1200 | 3 | 25.808 | 25.259-26.523 | 0.1721 | 5.952e+05 | 16.660 |
| 2 | 2 | 64x32 | 204,800 | 600 | 3 | 29.578 | 29.165-30.274 | 0.1972 | 1.039e+06 | 18.212 |
| 4 | 4 | 64x64 | 409,600 | 600 | 3 | 32.676 | 32.661-32.705 | 0.2178 | 1.880e+06 | 19.649 |
| 8 | 8 | 128x64 | 819,200 | 300 | 3 | 43.021 | 42.870-43.262 | 0.2868 | 2.856e+06 | 24.909 |
| 16 | 16 | 128x128 | 1,638,400 | 300 | 3 | 69.498 | 69.277-69.609 | 0.4633 | 3.536e+06 | 39.365 |
| 32 | 32 | 256x128 | 3,276,800 | 150 | 3 | 70.789 | 70.540-70.927 | 0.4719 | 6.943e+06 | 39.147 |
| 64 | 64 | 256x256 | 6,553,600 | 150 | 3 | 74.382 | 74.052-74.584 | 0.4959 | 1.322e+07 | 39.650 |
| 128 | 128 | 512x256 | 13,107,200 | 75 | 3 | 76.633 | 76.473-76.792 | 0.5109 | 2.566e+07 | 38.698 |
| 256 | 128 | 512x512 | 26,214,400 | 75 | 3 | 154.509 | 154.046-155.362 | 1.0301 | 2.545e+07 | 78.027 |
| 512 | 128 | 1024x512 | 52,428,800 | 37 | 3 | 411.817 | 410.058-412.730 | 2.7454 | 1.910e+07 | 219.1 |
| 1024 | 128 | 1024x1024 | 104,857,600 | 37 | 3 | 812.177 | 811.871-812.436 | 5.4145 | 1.937e+07 | 429.0 |

### dev_turbo_GPU (dt_G)

| i | ranks | NI x NJ | gridpoints | dt | runs | main loop (s) | spread (s) | s/step | throughput (cell-up/s) | continuity (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 32x32 | 102,400 | 1200 | 3 | 17.854 | 17.250-18.957 | 0.1190 | 8.603e+05 | 1.056 |
| 2 | 1 | 64x32 | 204,800 | 600 | 3 | 18.077 | 17.983-18.160 | 0.1205 | 1.699e+06 | 1.151 |
| 4 | 1 | 64x64 | 409,600 | 600 | 3 | 19.890 | 19.719-20.027 | 0.1326 | 3.089e+06 | 1.462 |
| 8 | 1 | 128x64 | 819,200 | 300 | 3 | 21.809 | 21.800-21.826 | 0.1454 | 5.634e+06 | 1.963 |
| 16 | 1 | 128x128 | 1,638,400 | 300 | 3 | 26.358 | 26.188-26.472 | 0.1757 | 9.324e+06 | 2.172 |
| 32 | 1 | 256x128 | 3,276,800 | 150 | 3 | 35.372 | 35.305-35.457 | 0.2358 | 1.390e+07 | 3.467 |
| 64 | 1 | 256x256 | 6,553,600 | 150 | 3 | 54.349 | 54.243-54.463 | 0.3623 | 1.809e+07 | 5.315 |
| 128 | 1 | 512x256 | 13,107,200 | 75 | 3 | 92.963 | 92.761-93.239 | 0.6198 | 2.115e+07 | 9.611 |
| 256 | 1 | 512x512 | 26,214,400 | 75 | 3 | 164.645 | 163.992-164.991 | 1.0976 | 2.388e+07 | 16.681 |
| 512 | 1 | 1024x512 | 52,428,800 | 37 | 3 | 395.364 | 392.043-400.900 | 2.6358 | 1.989e+07 | 47.703 |

### iturbo_GPU_amrex (it_G_ax)

| i | ranks | NI x NJ | gridpoints | dt | runs | main loop (s) | spread (s) | s/step | throughput (cell-up/s) | continuity (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 32x32 | 102,400 | 1200 | 3 | 18.963 | 18.720-19.301 | 0.1264 | 8.100e+05 | 2.552 |
| 2 | 1 | 64x32 | 204,800 | 600 | 3 | 20.244 | 20.174-20.329 | 0.1350 | 1.517e+06 | 3.363 |
| 4 | 1 | 64x64 | 409,600 | 600 | 3 | 23.208 | 22.866-23.430 | 0.1547 | 2.647e+06 | 5.060 |
| 8 | 1 | 128x64 | 819,200 | 300 | 3 | 26.586 | 26.437-26.829 | 0.1772 | 4.622e+06 | 7.934 |
| 16 | 1 | 128x128 | 1,638,400 | 300 | 3 | 35.888 | 35.835-35.983 | 0.2393 | 6.848e+06 | 12.895 |
| 32 | 1 | 256x128 | 3,276,800 | 150 | 3 | 54.032 | 53.616-54.529 | 0.3602 | 9.097e+06 | 24.278 |
| 64 | 1 | 256x256 | 6,553,600 | 150 | 3 | 91.856 | 91.346-92.341 | 0.6124 | 1.070e+07 | 46.114 |
| 128 | 1 | 512x256 | 13,107,200 | 75 | 3 | 166.351 | 163.341-168.302 | 1.1090 | 1.182e+07 | 90.252 |
| 256 | 1 | 512x512 | 26,214,400 | 75 | 3 | 325.621 | 319.280-334.128 | 2.1708 | 1.208e+07 | 176.8 |
| 512 | 1 | 1024x512 | 52,428,800 | 37 | 2 | 875.126 | 871.426-878.826 | 5.8342 | 8.986e+06 | 529.9 |

## Provenance


### Stack: dev-turbo-cpu

- **turbo-stack:** `2524b9d-dirty` (dirty working tree) (`/glade/work/altuntas/turbo-stack-dev-turbo-cpu`)
- **MOM6 submodule:** `ulm-10623-g108388fb6` (`108388fb608d8b861232bc203fac21ab7bc8f28b`)
- **GPU build flags** (ncar-nvhpc.mk):
  ```make
  FPPFLAGS := $(shell pkg-config --cflags yaml-0.1) -DHAVE_FC_DO_CONCURRENT_LOCAL
  FFLAGS += -mp=gpu -gpu=cc80,mem:separate -stdpar=gpu -Minfo=accel
  CFLAGS += -mp=gpu -gpu=cc80,mem:separate
  ```
- **Submodule snapshot:**
  ```
  f6466d899b66198593d6d40b3e8ca3dcbd343d8b dev-utils/gcovlens (heads/main)
   2c04fb23d0ee9ceef6d61f1021652ccab62e8324 submodules/MARBL (marbl0.48.2)
  +108388fb608d8b861232bc203fac21ab7bc8f28b submodules/MOM6 (ulm-10623-g108388fb6)
   6dd6d69bdb7c9efd4e210e1c459a897d1b02d21f submodules/amrex (25.11)
   7e526687b96ca685100f73edf7ef49214d5d5a19 submodules/infra/FMS2 (heads/dev/turbo)
   1647f85f695cd8f288b6471a99a078f48226efc0 submodules/infra/TIM (1647f85)
   12ac400e141854b54e5ce08c27c3301ef7d80074 submodules/pFUnit (v4.16.0-31-g12ac400)
  ```

> **Warning:** the turbo-stack working tree had uncommitted changes when this report was generated, so the commit hash does not fully capture the build. The GPU build flags above are recorded explicitly for this reason.

### Stack: iturbo-cpu

- **turbo-stack:** `fabef3b-dirty` (dirty working tree) (`/glade/work/altuntas/turbo-stack-iturbo-cpu`)
- **MOM6 submodule:** `ulm-10626-gd88ea2c91` (`d88ea2c9110b841705c3274dfe41cb1b5d1b4173`)
- **GPU build flags** (ncar-nvhpc.mk):
  ```make
  FPPFLAGS := $(shell pkg-config --cflags yaml-0.1) -DHAVE_FC_DO_CONCURRENT_LOCAL
  FFLAGS += -mp=gpu -gpu=cc80,mem:separate -stdpar=gpu -Minfo=accel
  CFLAGS += -mp=gpu -gpu=cc80,mem:separate
  ```
- **Submodule snapshot:**
  ```
  f6466d899b66198593d6d40b3e8ca3dcbd343d8b dev-utils/gcovlens (heads/main)
   2c04fb23d0ee9ceef6d61f1021652ccab62e8324 submodules/MARBL (marbl0.48.2)
  +d88ea2c9110b841705c3274dfe41cb1b5d1b4173 submodules/MOM6 (ulm-10626-gd88ea2c91)
   6dd6d69bdb7c9efd4e210e1c459a897d1b02d21f submodules/amrex (25.11)
   7e526687b96ca685100f73edf7ef49214d5d5a19 submodules/infra/FMS2 (heads/dev/turbo)
  +e94bbdde6074a57eb293b2cc95f6af47a6d8a0c7 submodules/infra/TIM (heads/main)
   12ac400e141854b54e5ce08c27c3301ef7d80074 submodules/pFUnit (v4.16.0-31-g12ac400)
  ```

> **Warning:** the turbo-stack working tree had uncommitted changes when this report was generated, so the commit hash does not fully capture the build. The GPU build flags above are recorded explicitly for this reason.

### Stack: dev-turbo

- **turbo-stack:** `2524b9d-dirty` (dirty working tree) (`/glade/work/altuntas/turbo-stack-dev-turbo`)
- **MOM6 submodule:** `ulm-10623-g108388fb6` (`108388fb608d8b861232bc203fac21ab7bc8f28b`)
- **GPU build flags** (ncar-nvhpc.mk):
  ```make
  FPPFLAGS := $(shell pkg-config --cflags yaml-0.1) -DHAVE_FC_DO_CONCURRENT_LOCAL
  FFLAGS += -mp=gpu -gpu=cc80,mem:separate -stdpar=gpu -Minfo=accel
  CFLAGS += -mp=gpu -gpu=cc80,mem:separate
  ```
- **Submodule snapshot:**
  ```
  f6466d899b66198593d6d40b3e8ca3dcbd343d8b dev-utils/gcovlens (heads/main)
   2c04fb23d0ee9ceef6d61f1021652ccab62e8324 submodules/MARBL (marbl0.48.2)
  +108388fb608d8b861232bc203fac21ab7bc8f28b submodules/MOM6 (ulm-10623-g108388fb6)
   6dd6d69bdb7c9efd4e210e1c459a897d1b02d21f submodules/amrex (25.11)
   7e526687b96ca685100f73edf7ef49214d5d5a19 submodules/infra/FMS2 (heads/dev/turbo)
   1647f85f695cd8f288b6471a99a078f48226efc0 submodules/infra/TIM (1647f85)
   12ac400e141854b54e5ce08c27c3301ef7d80074 submodules/pFUnit (v4.16.0-31-g12ac400)
  ```

> **Warning:** the turbo-stack working tree had uncommitted changes when this report was generated, so the commit hash does not fully capture the build. The GPU build flags above are recorded explicitly for this reason.

### Stack: iturbo

- **turbo-stack:** `81e64a1-dirty` (dirty working tree) (`/glade/work/altuntas/turbo-stack-iturbo`)
- **MOM6 submodule:** `ulm-10627-g1a8b32aa8` (`1a8b32aa89c04a2903502f6268effcfb746279d9`)
- **GPU build flags** (ncar-nvhpc.mk):
  ```make
  FPPFLAGS := $(shell pkg-config --cflags yaml-0.1) -DHAVE_FC_DO_CONCURRENT_LOCAL
  FFLAGS += -mp=gpu -gpu=cc80,mem:separate -stdpar=gpu -Minfo=accel
  CFLAGS += -mp=gpu -gpu=cc80,mem:separate
  ```
- **Submodule snapshot:**
  ```
  f6466d899b66198593d6d40b3e8ca3dcbd343d8b dev-utils/gcovlens (heads/main)
   2c04fb23d0ee9ceef6d61f1021652ccab62e8324 submodules/MARBL (marbl0.48.2)
  +1a8b32aa89c04a2903502f6268effcfb746279d9 submodules/MOM6 (ulm-10627-g1a8b32aa8)
   6dd6d69bdb7c9efd4e210e1c459a897d1b02d21f submodules/amrex (25.11)
   7e526687b96ca685100f73edf7ef49214d5d5a19 submodules/infra/FMS2 (heads/dev/turbo)
  +b3a1315ae319d437e2745dd099d6bb40ede085fd submodules/infra/TIM (heads/docs/io-diag-pio-roadmap)
   12ac400e141854b54e5ce08c27c3301ef7d80074 submodules/pFUnit (v4.16.0-31-g12ac400)
  ```

> **Warning:** the turbo-stack working tree had uncommitted changes when this report was generated, so the commit hash does not fully capture the build. The GPU build flags above are recorded explicitly for this reason.


