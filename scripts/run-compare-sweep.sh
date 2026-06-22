#!/bin/bash
# MOM6 double_gyre four-config comparison sweep on Derecho.
#
# Usage: sh run-compare-sweep.sh <config> [jobsizes] [nrepeats]
#
#   config    one of: dev_turbo_CPU iturbo_CPU_amrex
#                     dev_turbo_GPU iturbo_GPU_amrex
#   jobsizes  space-separated job-size indices (default "1 2 ... 1024")
#   nrepeats  repeat runs per (config, size) point (default 3)
#
# One config per invocation, so a PBS job can run any subset. Each config is an
# executable from a specific turbo-stack checkout plus, for the *_amrex
# variants, the six *_MODE=AMREX environment variables that route the continuity
# PPM kernels through AMReX. Stack roots are overridable via STACK_DEV_TURBO_CPU,
# STACK_ITURBO_CPU, STACK_DEV_TURBO, STACK_ITURBO.
#
# The sweep itself matches run-scaling-sweep.sh: job-size index i sets a
# near-square layout of i 32x32 column blocks at NK=100, and every run advances
# exactly 150 dynamic steps (TIMEUNIT = dt with DAYMAX = 150) so wall-clock is
# comparable across sizes and configs. CPU configs weak-scale (ranks = min(i,
# 128)); GPU configs run 1 rank on 1 GPU. Run from a double_gyre run directory
# (MOM_input, input.nml with clock_grain='ROUTINE', diag_table). Produces
# <config>_<i0>_run<r>.out/.err/.stats per repeat for gen_compare_report.py.

# Shared helpers (module load, CONFIG->STACK/exec, AMReX env, get_layout,
# MOM_override) live in lib-compare.sh next to this script.
. "$(dirname "$0")/lib-compare.sh"
load_modules

CONFIG=${1:?usage: sh run-compare-sweep.sh <config> [jobsizes] [nrepeats]}
JOBSIZES=${2:-"1 2 4 8 16 32 64 128 256 512 1024"}
NRUNS=${3:-3}

# Sets STACK, PLATFORM and MOM6_EXEC (stack roots overridable via STACK_*).
resolve_stack "${CONFIG}"
set_amrex_env "${CONFIG}"

# AMReX/TinyProfiler args (iturbo builds; harmless/ignored for dev/turbo):
#  - the_arena_init_size=0: don't pre-reserve 3/4 of GPU memory for AMReX's
#    arena (the default), which otherwise starves MOM6's Fortran stdpar
#    allocations and OOMs the GPU at ~4x smaller domains than dev/turbo. With
#    0 the arena grows on demand to its (small) working set; steady-state perf
#    is unchanged (alloc/free is ~0.06% of continuity cost).
#  - tiny_profiler.*: synchronize around regions so kernel timers are
#    wall-clock comparable, and print all rows.
TINY_ARGS="amrex.the_arena_init_size=0 tiny_profiler.device_synchronize_around_region=1 tiny_profiler.print_threshold=0 amrex.the_arena_is_managed=0"

CPU_PER_NODE=128

export NGPUS=1  # exported so set_gpu_rank can read it in each rank

#---

for i in ${JOBSIZES}; do
    nranks=$(( i > CPU_PER_NODE ? CPU_PER_NODE : i ))

    # GPU runs on NGPUS ranks (1 -> 1x1 layout); CPU weak-scales to nranks.
    if [ "$PLATFORM" = "gpu" ]; then
	   rank_count=${NGPUS}
    else
	   rank_count=${nranks}
    fi

    write_mom_override "${i}" "${rank_count}" 150
    # File label index
    printf -v i0 "%03d" "$i"

    for ((r = 1; r <= NRUNS; r++)); do
        echo " "
        echo "=== ${CONFIG}: size i=${i}, run ${r} of ${NRUNS} ==="
        echo " "
        TAG=${CONFIG}_${i0}_run${r}

        if [ "${PLATFORM}" == "gpu" ]; then
            # set_gpu_rank handles the per-rank CUDA_VISIBLE_DEVICES assignment;
            # CPU placement/affinity stays on the mpirun flags below.
            # Redirect mpiexec's OWN stderr to .err *before* the pipe; a trailing
            # `2>` after `| tee` would capture tee's stderr (always empty), losing
            # the model's CUDA/OOM/abort messages. stdout still tees to .out.
            mpiexec -np ${NGPUS} \
                --ppn 1 \
                --cpu-bind=core \
                set_gpu_rank ${MOM6_EXEC} ${TINY_ARGS} \
                2> ${TAG}.err \
                | tee ${TAG}.out
        else
            # For socket-level binding instead of core, use --cpu-bind=socket.
            mpiexec -np ${nranks} \
                --ppn ${nranks} \
                --cpu-bind=core \
                ${MOM6_EXEC} ${TINY_ARGS} \
                2> ${TAG}.err \
                | tee ${TAG}.out
        fi

        # ocean.stats is the per-run diagnostic checksum file; keep it so the
        # report can cross-check that the configs computed the same physics.
        [ -f ocean.stats ] && mv ocean.stats ${TAG}.stats
    done
done
