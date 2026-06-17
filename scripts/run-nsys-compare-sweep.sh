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

module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3

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

# Default GPU stack checkouts per build variant (override via env).
STACK_DEV_TURBO=${STACK_DEV_TURBO:-/glade/work/altuntas/turbo-stack-dev-turbo}
STACK_ITURBO=${STACK_ITURBO:-/glade/work/altuntas/turbo-stack-iturbo}

case "${CONFIG}" in
    dev_turbo_GPU)    STACK=${STACK_DEV_TURBO} ;;
    iturbo_GPU_amrex) STACK=${STACK_ITURBO} ;;
    dev_turbo_CPU|iturbo_CPU_amrex)
        echo "run-nsys-compare-sweep.sh: ${CONFIG} is a CPU config; profiling is" >&2
        echo "  GPU-only (per-kernel device timers come from the GPU trace)." >&2
        echo "  Use run-compare-sweep.sh for the CPU configs." >&2
        exit 1 ;;
    *) echo "run-nsys-compare-sweep.sh: unknown config: ${CONFIG}" >&2; exit 1 ;;
esac

MOM6_EXEC=${STACK}/bin/nvhpc/MOM6_using_TIM/MOM6/MOM6
if [ ! -x "${MOM6_EXEC}" ]; then
    echo "run-nsys-compare-sweep.sh: MOM6 executable not found or not executable: ${MOM6_EXEC}" >&2
    echo "  check the stack for config ${CONFIG} (got: ${STACK})" >&2
    exit 1
fi

# amrex configs route the six ported PPM kernels through AMReX; others unset them
# (the same executable serves both modes).
if [ "${CONFIG#*amrex}" != "${CONFIG}" ]; then
    export ZONAL_EDGE_THICKNESS_MODE=AMREX
    export MERIDIONAL_EDGE_THICKNESS_MODE=AMREX
    export PPM_LIMIT_POS_MODE=AMREX
    export PPM_LIMIT_CW84_MODE=AMREX
    export PPM_RECONSTRUCTION_X_MODE=AMREX
    export PPM_RECONSTRUCTION_Y_MODE=AMREX
else
    unset ZONAL_EDGE_THICKNESS_MODE
    unset MERIDIONAL_EDGE_THICKNESS_MODE
    unset PPM_LIMIT_POS_MODE
    unset PPM_LIMIT_CW84_MODE
    unset PPM_RECONSTRUCTION_X_MODE
    unset PPM_RECONSTRUCTION_Y_MODE
fi

# iturbo args (harmless for dev/turbo): arena_init_size=0 stops AMReX reserving
# 3/4 of GPU memory (else it OOMs the Fortran stdpar allocs); device_synchronize
# makes kernels wall-clock attributable with clean Nsight boundaries.
TINY_ARGS="amrex.the_arena_init_size=0 tiny_profiler.device_synchronize_around_region=1 tiny_profiler.print_threshold=0"

export NGPUS=1  # so set_gpu_rank can read it

#---

# Construct a square-like layout m x n for i ranks
get_layout() {
 	local i=$1
  	m=1

	# Find the smallest m such that m**2 > i
  	while (( (m+1)*(m+1) <= i )); do
    	((m++))
  	done

	# Then decrement m until it exactly divides i
  	while (( i % m != 0 )); do
  	 ((m--))
  	done

	# Finally, set n such that m*n == i
  	n=$(( i / m ))

	# Force m >= n
  	if (( m < n )); then
  	 	local t=$m
  	 	m=$n
  	 	n=$t
  	fi
}

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

    cat <<EOF > MOM_override
#override COORD_CONFIG = "linear"
DENSITY_RANGE = 2.0
#override NK = 100
#override NIGLOBAL = ${ni}
#override NJGLOBAL = ${nj}
LAYOUT = ${lx},${ly}
#override DT = ${dt}
#override DT_THERM = ${dt_therm}
#override DT_FORCING = ${dt_therm}
TIMEUNIT = ${dt}
ENERGYSAVEDAYS = 50
#override DAYMAX = ${NSTEPS}
#override READ_DEPTH_LIST = False
#override RESTART_CONTROL = -1
EOF
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
