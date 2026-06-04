#!/bin/bash
# One-off Nsight Systems profiling run of the double_gyre GPU test.
#
# Usage: sh run-profile.sh [jobsize]     (default: 32)
#
# Run from a double_gyre run directory (MOM_input, input.nml, diag_table).
# Produces prof_gpu_<i>.nsys-rep (open with `nsys-ui` or `nsys stats <file>`)
# plus the kernel/memcpy/API summary tables in prof_gpu_<i>.out.
#
# Same domain/timestep setup as run-scaling-sweep.sh (GPU branch, 1 rank on 1 GPU),
# except DAYMAX: 20 steps instead of 150 -- enough to see steady-state
# per-step behavior without a huge trace.

module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3

# Resolve the stack. This script lives in <repo>/scripts/, so default
# TURBO_STACK to a sibling 'turbo-stack-for-prof' checkout next to the repo.
# Override TURBO_STACK in the environment to point elsewhere.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TURBO_STACK="${TURBO_STACK:-$(dirname "$(dirname "$SCRIPT_DIR")")/turbo-stack-for-prof}"
MOM6_EXEC=${TURBO_STACK}/bin/nvhpc/MOM6_using_TIM/MOM6/MOM6

i=${1:-32}

export NGPUS=1  # exported so set_gpu_rank can read it in each rank
RANKS_PER_NODE=1

NSTEPS=20

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
EOF

# File label index
printf -v i0 "%03d" "$i"

# set_gpu_rank sets CUDA_VISIBLE_DEVICES, then execs nsys, which launches MOM6.
# --stats=true prints the CUDA kernel / memcpy / API summary tables on exit:
# they split time into device compute vs host<->device transfers vs host gaps.
mpiexec -np ${NGPUS} \
    --ppn ${RANKS_PER_NODE} \
    --cpu-bind=core \
    set_gpu_rank \
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --stats=true \
        --force-overwrite=true \
        -o prof_gpu_${i0} \
        ${MOM6_EXEC} \
    | tee prof_gpu_${i0}.out 2> prof_gpu_${i0}.err
