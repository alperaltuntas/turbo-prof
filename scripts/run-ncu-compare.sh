#!/bin/bash
# MOM6 double_gyre comparison run UNDER NSIGHT COMPUTE (GPU configs only).
#
# Usage: sh run-ncu-compare.sh <config> [i] [nsteps]
#   config    dev_turbo_GPU or iturbo_GPU_amrex   (GPU only)
#   i         single job-size index               (default 16)
#   nsteps    dynamic steps to advance            (default 2)
#
# Nsight Compute counterpart to run-nsys-compare-sweep.sh. nsys traces a whole
# run cheaply to split GPU vs copy time; ncu instead REPLAYS each kernel launch
# many times to collect the full hardware-counter set (occupancy, memory
# throughput, stalls, ...), so it is far slower and is scoped to ONE size and a
# couple of steps rather than a sweep. Writes a single full report
# ncu_<config>_<i0>.{ncu-rep,out,err} (the .ncu-rep opens in the Nsight Compute
# UI / `ncu-ui`), plus a details-page CSV next to it (disable with DUMP_CSV=0).
# GPU only (CPU configs error). `ncu_` prefix avoids colliding with the plain
# and nsys sweeps' logs in the same dir.
#
# ncu profiles EVERY kernel launch in the run by default; with --set full and
# MOM6's many kernels even two steps takes a long time. Scope it down with a
# small i, a kernel-name filter (NCU_KERNEL_FILTER, ncu -k regex), or by
# skipping warmup / capping launches (LAUNCH_SKIP / LAUNCH_COUNT). NSTEPS
# defaults to 2 so the run reaches steady state without flooding the report.
#
# Run from a double_gyre run dir (input.nml with clock_grain='ROUTINE') on a GPU
# node. The recorder's ncu comes from nvhpc/25.9 (the module load puts it on
# PATH). Env: STACK_DEV_TURBO/STACK_ITURBO, NSTEPS, NCU, NCU_SET,
# NCU_KERNEL_FILTER, LAUNCH_SKIP, LAUNCH_COUNT, DUMP_CSV.

# Shared helpers (module load, CONFIG->STACK/exec, AMReX env, get_layout,
# MOM_override) live in lib-compare.sh next to this script.
. "$(dirname "$0")/lib-compare.sh"
load_modules

CONFIG=${1:?usage: sh run-ncu-compare.sh <config> [i] [nsteps]}
I=${2:-16}
NSTEPS=${3:-${NSTEPS:-2}}

NCU=${NCU:-ncu}
# Metric set to collect. `full` is every section (the "full ncu report"); narrow
# to e.g. `basic` or a specific section for faster, smaller captures.
NCU_SET=${NCU_SET:-full}
# Optional ncu -k kernel-name regex (empty = profile all kernels).
NCU_KERNEL_FILTER=${NCU_KERNEL_FILTER:-}
# Launch window: skip the first LAUNCH_SKIP kernel launches (warmup), then
# profile LAUNCH_COUNT of them (empty = no cap, profile to end of the run).
LAUNCH_SKIP=${LAUNCH_SKIP:-0}
LAUNCH_COUNT=${LAUNCH_COUNT:-}
DUMP_CSV=${DUMP_CSV:-1}

# Sets STACK and MOM6_EXEC (stack roots overridable via STACK_*); GPU-only, so
# reject CPU configs up front.
resolve_stack "${CONFIG}"
require_gpu_config "${CONFIG}"
set_amrex_env "${CONFIG}"

# iturbo args (harmless for dev/turbo): arena_init_size=0 stops AMReX reserving
# 3/4 of GPU memory (else it OOMs the Fortran stdpar allocs). ncu serializes
# kernels itself, so no extra device_synchronize is needed here.
TINY_ARGS="amrex.the_arena_init_size=0 tiny_profiler.print_threshold=0"

export NGPUS=1  # so set_gpu_rank can read it

#---

echo "=== ncu-compare run: ${CONFIG} @ $(date) ==="
echo "    stack:  ${STACK}"
echo "    size:   i=${I}"
echo "    steps:  ${NSTEPS}"
echo "    ncu:    ${NCU} (set=${NCU_SET})"
[ -n "${NCU_KERNEL_FILTER}" ] && echo "    kernel: -k '${NCU_KERNEL_FILTER}'"
echo "    launch: skip=${LAUNCH_SKIP} count=${LAUNCH_COUNT:-<all>}"

# GPU: 1 rank on 1 device, 1x1 layout.
get_layout "${NGPUS}"
lx=${m}
ly=${n}

get_layout "${I}"
ni=$(( 32 * ${m} ))
nj=$(( 32 * ${n} ))
dt=$(( 1200 / ${m} ))
dt_therm=$(( 2400 / ${m} ))

write_mom_override "${NSTEPS}"
printf -v i0 "%03d" "${I}"

TAG=ncu_${CONFIG}_${i0}

# Optional ncu launch-window / kernel-filter flags, only when requested.
NCU_OPTS=""
[ "${LAUNCH_SKIP}" != "0" ] && NCU_OPTS="${NCU_OPTS} --launch-skip ${LAUNCH_SKIP}"
[ -n "${LAUNCH_COUNT}" ]    && NCU_OPTS="${NCU_OPTS} --launch-count ${LAUNCH_COUNT}"
[ -n "${NCU_KERNEL_FILTER}" ] && NCU_OPTS="${NCU_OPTS} -k ${NCU_KERNEL_FILTER}"

echo " "
echo "=== ${CONFIG} [ncu]: size i=${I} @ $(date) ==="
echo " "

# set_gpu_rank -> ncu -> MOM6 (as run-nsys-compare-sweep.sh, but ncu has no
# `profile` subcommand). --target-processes all so ncu follows through the
# set_gpu_rank wrapper to MOM6. -o writes ${TAG}.ncu-rep; --force-overwrite
# replaces a prior one. The 2> goes before the pipe so MOM6's CUDA/OOM messages
# land in .err, not tee's.
mpiexec -np ${NGPUS} \
    --ppn 1 \
    --cpu-bind=core \
    set_gpu_rank \
    ${NCU} \
        --set ${NCU_SET} \
        --target-processes all \
        ${NCU_OPTS} \
        --force-overwrite \
        -o ${TAG} \
        ${MOM6_EXEC} ${TINY_ARGS} \
    2> ${TAG}.err \
    | tee ${TAG}.out

# Keep the per-run diagnostic checksum file alongside the report.
[ -f ocean.stats ] && mv ocean.stats ${TAG}.stats

# Dump the details page to CSV next to the .ncu-rep (full untruncated metric
# names; the .out text page clips them and the .ncu-rep needs the UI). Reuses the
# report ncu just wrote. Skip if the run produced no report (e.g. an OOM), or if
# DUMP_CSV=0.
if [ "${DUMP_CSV}" != "0" ] && [ -f "${TAG}.ncu-rep" ]; then
    echo "  dumping CSV report: ${TAG}.csv"
    ${NCU} --import "${TAG}.ncu-rep" --page details --csv > "${TAG}.csv" 2>/dev/null
fi

echo "=== ncu-compare run COMPLETE: ${CONFIG} @ $(date) ==="
