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

module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3

CONFIG=${1:?usage: sh run-compare-sweep.sh <config> [jobsizes] [nrepeats]}
JOBSIZES=${2:-"1 2 4 8 16 32 64 128 256 512 1024"}
NRUNS=${3:-3}

# Default stack checkouts per build variant (override via env).
STACK_DEV_TURBO_CPU=${STACK_DEV_TURBO_CPU:-/glade/work/altuntas/turbo-stack-dev-turbo-cpu}
STACK_ITURBO_CPU=${STACK_ITURBO_CPU:-/glade/work/altuntas/turbo-stack-iturbo-cpu}
STACK_DEV_TURBO=${STACK_DEV_TURBO:-/glade/work/altuntas/turbo-stack-dev-turbo}
STACK_ITURBO=${STACK_ITURBO:-/glade/work/altuntas/turbo-stack-iturbo}

case "${CONFIG}" in
    dev_turbo_CPU)    STACK=${STACK_DEV_TURBO_CPU} ;;
    iturbo_CPU_amrex) STACK=${STACK_ITURBO_CPU} ;;
    dev_turbo_GPU)    STACK=${STACK_DEV_TURBO} ;;
    iturbo_GPU_amrex) STACK=${STACK_ITURBO} ;;
    *) echo "run-compare-sweep.sh: unknown config: ${CONFIG}" >&2; exit 1 ;;
esac
case "${CONFIG}" in
    *CPU*) PLATFORM=cpu ;;
    *)     PLATFORM=gpu ;;
esac

MOM6_EXEC=${STACK}/bin/nvhpc/MOM6_using_TIM/MOM6/MOM6
if [ ! -x "${MOM6_EXEC}" ]; then
    echo "run-compare-sweep.sh: MOM6 executable not found or not executable: ${MOM6_EXEC}" >&2
    echo "  check the stack for config ${CONFIG} (got: ${STACK})" >&2
    exit 1
fi

# AMReX configs route the six ported PPM kernels through AMReX; all other
# configs must run with these unset (the same executable serves both modes).
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

# AMReX/TinyProfiler args (iturbo builds; harmless/ignored for dev/turbo):
#  - the_arena_init_size=0: don't pre-reserve 3/4 of GPU memory for AMReX's
#    arena (the default), which otherwise starves MOM6's Fortran stdpar
#    allocations and OOMs the GPU at ~4x smaller domains than dev/turbo. With
#    0 the arena grows on demand to its (small) working set; steady-state perf
#    is unchanged (alloc/free is ~0.06% of continuity cost).
#  - tiny_profiler.*: synchronize around regions so kernel timers are
#    wall-clock comparable, and print all rows.
TINY_ARGS="amrex.the_arena_init_size=0 tiny_profiler.device_synchronize_around_region=1 tiny_profiler.print_threshold=0"

CPU_PER_NODE=128

export NGPUS=1  # exported so set_gpu_rank can read it in each rank

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

for i in ${JOBSIZES}; do
    nranks=$(( i > CPU_PER_NODE ? CPU_PER_NODE : i ))

    if [ "$PLATFORM" = "gpu" ]; then
	   get_layout "${NGPUS}"
    else
	   get_layout "${nranks}"
    fi
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
#override DAYMAX = 150
#override READ_DEPTH_LIST = False
#override RESTART_CONTROL = -1
EOF
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
