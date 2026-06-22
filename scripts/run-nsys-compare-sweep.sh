#!/bin/bash
# MOM6 double_gyre comparison sweep UNDER NSIGHT SYSTEMS (GPU configs only).
#
# Usage: sh run-nsys-compare-sweep.sh <config> [jobsizes] [nrepeats]
#   config    dev_turbo_GPU or iturbo_GPU_amrex   (GPU only)
#   jobsizes  job-size indices (default "1 2 ... 1024")
#   nrepeats  repeats per (config, size)          (default 2)
#
# Nsight counterpart to run-compare-sweep.sh -- a separate script (as run-profile.sh
# is to run-scaling-sweep.sh) so the plain sweep's producer stays untouched. Same
# GPU runs, each wrapped in `nsys profile --stats` to split continuity cost into
# GPU compute vs host<->device copies. Writes
# prof_<config>_<i0>_run<r>.{nsys-rep,out,err} for gen_nsys_compare_report.py,
# plus per-trace CSV reports (full, untruncated kernel/memcpy/API/NVTX names the
# .out text tables clip) -- disable with DUMP_CSV=0.
# GPU only (CPU configs error); NSTEPS defaults to 40 (vs 150) for small traces;
# `prof_` prefix avoids colliding with the plain sweep's logs in the same dir.
#
# Run from a double_gyre run dir (input.nml with clock_grain='ROUTINE') on a GPU
# node. Re-processing the trace needs the recorder's nsys (nvhpc/25.9, >=2025.5;
# the module load puts it on PATH). Env: STACK_DEV_TURBO/STACK_ITURBO, NSTEPS,
# NSYS, DUMP_CSV.

# Shared helpers (module load, CONFIG->STACK/exec, AMReX env, get_layout,
# MOM_override) live in lib-compare.sh next to this script.
. "$(dirname "$0")/lib-compare.sh"
load_modules

CONFIG=${1:?usage: sh run-nsys-compare-sweep.sh <config> [jobsizes] [nrepeats]}
JOBSIZES=${2:-"1 2 4 8 16 32 64 128 256 512"}
NRUNS=${3:-2}

NSTEPS=${NSTEPS:-40}
NSYS=${NSYS:-nsys}
DUMP_CSV=${DUMP_CSV:-1}
# Per-trace CSV report to dump (auto-demangled): only cuda_gpu_kern_sum, the one
# gen_nsys_compare_report.py consumes (leaf compute totals/ratio/per-launch). The
# .nsys-rep still captures everything, so other reports can be extracted ad hoc.
CSV_REPORTS="cuda_gpu_kern_sum"

# Sets STACK and MOM6_EXEC (stack roots overridable via STACK_*); GPU-only, so
# reject CPU configs up front.
resolve_stack "${CONFIG}"
require_gpu_config "${CONFIG}"
set_amrex_env "${CONFIG}"

# iturbo args (harmless for dev/turbo): arena_init_size=0 stops AMReX reserving
# 3/4 of GPU memory (else it OOMs the Fortran stdpar allocs); device_synchronize
# makes kernels wall-clock attributable with clean Nsight boundaries.
TINY_ARGS="amrex.the_arena_init_size=0 tiny_profiler.device_synchronize_around_region=1 tiny_profiler.print_threshold=0"

export NGPUS=1  # so set_gpu_rank can read it

#---

echo "=== nsys-compare sweep: ${CONFIG} @ $(date) ==="
echo "    stack:  ${STACK}"
echo "    sizes:  ${JOBSIZES}"
echo "    steps:  ${NSTEPS}   repeats: ${NRUNS}"
echo "    nsys:   ${NSYS}"

for i in ${JOBSIZES}; do
    # GPU: 1 rank on 1 device, 1x1 layout.
    get_layout "${NGPUS}"
    lx=${m}
    ly=${n}

    get_layout "${i}"
    ni=$(( 32 * ${m} ))
    nj=$(( 32 * ${n} ))
    dt=$(( 1200 / ${m} ))
    dt_therm=$(( 2400 / ${m} ))

    write_mom_override "${NSTEPS}"
    printf -v i0 "%03d" "$i"

    for ((r = 1; r <= NRUNS; r++)); do
        echo " "
        echo "=== ${CONFIG} [nsys]: size i=${i}, run ${r} of ${NRUNS} @ $(date) ==="
        echo " "
        TAG=prof_${CONFIG}_${i0}_run${r}

        # set_gpu_rank -> nsys -> MOM6 (as run-profile.sh). --stats prints the
        # kernel/memcpy/API tables into .out; -o writes ${TAG}.nsys-rep. The 2>
        # goes before the pipe so MOM6's CUDA/OOM messages land in .err, not tee's.
        mpiexec -np ${NGPUS} \
            --ppn 1 \
            --cpu-bind=core \
            set_gpu_rank \
            ${NSYS} profile \
                --trace=cuda,nvtx,osrt \
                --cuda-memory-usage=true \
                --stats=true \
                --force-overwrite=true \
                -o ${TAG} \
                ${MOM6_EXEC} ${TINY_ARGS} \
            2> ${TAG}.err \
            | tee ${TAG}.out

        # Keep the per-run diagnostic checksum file alongside the trace.
        [ -f ocean.stats ] && mv ocean.stats ${TAG}.stats

        # Dump per-trace CSV reports next to the .nsys-rep (full untruncated names;
        # the .out text tables clip them). Reuses the .sqlite the --stats export
        # just wrote, so no re-export. Writes ${TAG}_<report>.csv. Skip if the run
        # produced no trace (e.g. an OOM), or if DUMP_CSV=0.
        # NOTE: `nsys stats -o` SILENTLY SKIPS an output CSV that already exists, so
        # on a re-profile it would leave stale data in place -- remove any prior
        # CSVs for this trace first so they always reflect the current run.
        if [ "${DUMP_CSV}" != "0" ] && [ -f "${TAG}.nsys-rep" ]; then
            echo "  dumping CSV reports: ${TAG}_{${CSV_REPORTS// /,}}.csv"
            for rep in ${CSV_REPORTS}; do rm -f "${TAG}_${rep}.csv"; done
            ${NSYS} stats --format csv -o "${TAG}" \
                $(for rep in ${CSV_REPORTS}; do echo --report ${rep}; done) \
                "${TAG}.nsys-rep" >/dev/null 2>&1
        fi
    done
done

echo "=== nsys-compare sweep COMPLETE: ${CONFIG} @ $(date) ==="
