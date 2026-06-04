#!/bin/bash
# MOM6 double_gyre scaling sweep on Derecho.
#
# Usage: sh run-scaling-sweep.sh <cpu|gpu>
#
#   cpu  weak scaling then saturated node; FMS2 build; up to 128 ranks/node.
#   gpu  single-device problem-size scan; TIM (GPU-offload) build; 1 rank/GPU.
#
# Both branches advance exactly 150 dynamic steps per job size (TIMEUNIT = dt
# with DAYMAX = 150), so wall-clock is comparable across problem sizes. Run from
# a double_gyre run directory (MOM_input, input.nml, diag_table). Produces one
# <platform>_<i>.out per job size for gen_report.py to parse.

module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3

# Platform comes from the first argument and selects the matching build.
PLATFORM=${1:-}
case "${PLATFORM}" in
    cpu) MOM6_BUILD=MOM6_using_FMS2 ;;
    gpu) MOM6_BUILD=MOM6_using_TIM ;;
    *)   echo "usage: $0 <cpu|gpu>" >&2; exit 1 ;;
esac

# Resolve the stack. This script lives in <repo>/scripts/, so default
# TURBO_STACK to a sibling 'turbo-stack-for-prof' checkout next to the repo.
# Override TURBO_STACK in the environment to point elsewhere.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TURBO_STACK="${TURBO_STACK:-$(dirname "$(dirname "$SCRIPT_DIR")")/turbo-stack-for-prof}"
MOM6_EXEC=${TURBO_STACK}/bin/nvhpc/${MOM6_BUILD}/MOM6/MOM6

JOBSIZES="1 2 4 8 16 32 64 128 256 512 1024"

CPU_PER_NODE=128
RANKS_PER_NODE=1

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
EOF
    # File label index
    printf -v i0 "%03d" "$i"

    if [ "${PLATFORM}" == "gpu" ]; then
        # set_gpu_rank handles the per-rank CUDA_VISIBLE_DEVICES assignment;
        # CPU placement/affinity stays on the mpirun flags below.
        mpiexec -np ${NGPUS} \
            --ppn ${RANKS_PER_NODE} \
            --cpu-bind=core \
            set_gpu_rank ${MOM6_EXEC} \
            | tee ${PLATFORM}_${i0}.out 2> ${PLATFORM}_${i0}.err
    else
        # For socket-level binding instead of core, use --cpu-bind=socket.
        mpiexec -np ${nranks} \
            --ppn ${nranks} \
            --cpu-bind=core \
            ${MOM6_EXEC} \
            | tee ${PLATFORM}_${i0}.out 2> ${PLATFORM}_${i0}.err
    fi

done
