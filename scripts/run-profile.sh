#!/bin/bash
# One-off Nsight Systems profiling run of the double_gyre GPU test.
#
# Usage: sh run-profile.sh <turbo_stack> [jobsize] [kernel_mode]
#          turbo_stack  path to the turbo-stack checkout holding the build
#                       (its MOM6 lives at <turbo_stack>/bin/nvhpc/MOM6_using_TIM/
#                       MOM6/MOM6). REQUIRED -- no default; a wrong stack here is
#                       what silently produced empty traces before.
#          jobsize      job-size index i (default: 32)
#          kernel_mode  FORTRAN (default) or AMREX -- which path the ported
#                       continuity PPM kernels take. AMREX sets the six per-kernel
#                       *_MODE env vars so the AMReX (TIM) kernels run instead of
#                       the Fortran ones; this requires the MOM6_using_TIM build
#                       with a CUDA AMReX (see build-cuda-amrex.sh in the stack).
#
# Run from a double_gyre run directory (MOM_input, input.nml, diag_table).
# Produces prof_<mode>_<i>.nsys-rep (open with `nsys-ui` or `nsys stats <file>`)
# plus the kernel/memcpy/API summary tables in prof_<mode>_<i>.out. The point of
# the AMREX run is to read, off the Nsight tables, the AMReX continuity *kernel*
# time separately from the host<->device copies the bridge does around it (which
# the FMS mpp_clock timer folds together) -- i.e. the split the MOM timers can't
# give. Compare against the FORTRAN run at the same size.
#
# Same domain/timestep setup as run-scaling-sweep.sh (GPU branch, 1 rank on 1 GPU),
# except DAYMAX: 20 steps instead of 150 -- enough to see steady-state
# per-step behavior without a huge trace.

module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3

# Resolve the stack. TURBO_STACK is a REQUIRED first argument
TURBO_STACK=${1:?usage: sh run-profile.sh <turbo_stack> [jobsize] [kernel_mode]}
MOM6_EXEC=${TURBO_STACK}/bin/nvhpc/MOM6_using_TIM/MOM6/MOM6
if [ ! -x "${MOM6_EXEC}" ]; then
    echo "run-profile.sh: MOM6 executable not found or not executable: ${MOM6_EXEC}" >&2
    echo "  check the turbo_stack argument (got: ${TURBO_STACK})" >&2
    exit 1
fi

i=${2:-32}
KMODE=$(echo "${3:-FORTRAN}" | tr '[:lower:]' '[:upper:]')

export NGPUS=1  # exported so set_gpu_rank can read it in each rank
RANKS_PER_NODE=1

NSTEPS=150

# Select which path the ported continuity PPM kernels take. Each var defaults to
# FORTRAN inside MOM6 (getenv_mode); setting them to AMREX routes the call
# through the C++/AMReX bridge instead. All six together cover the ported set
# (edge thickness, PPM limiters, PPM reconstruction).
if [ "${KMODE}" = "AMREX" ]; then
    export ZONAL_EDGE_THICKNESS_MODE=AMREX
    export MERIDIONAL_EDGE_THICKNESS_MODE=AMREX
    export PPM_LIMIT_POS_MODE=AMREX
    export PPM_LIMIT_CW84_MODE=AMREX
    export PPM_RECONSTRUCTION_X_MODE=AMREX
    export PPM_RECONSTRUCTION_Y_MODE=AMREX
fi

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

# READ_DEPTH_LIST=False below disables the depth-list checksum during init. On the
# TIM build MOM6 field_checksum routes to TIM::checksum, which runs an
# amrex::Reduce::Sum with an AMREX_GPU_DEVICE lambda over the raw HOST field
# pointer -- fine on the CPU AMReX backend, but with the CUDA backend it launches a
# GPU kernel dereferencing host memory => illegal access (CUDA 700) in
# mom_sum_output init. READ_DEPTH_LIST=False takes depth_list_setup's else branch
# (create_depth_list only), which never checksums; with DEBUG off that is the only
# checksum in the run. Keep this comment in the SCRIPT, not in MOM_override -- MOM6
# parses '#' lines and an apostrophe there trips its "mismatched quote" parser.
# RESTART_CONTROL=-1 below skips the end-of-run restart write. MOM6 checksums every
# field as it saves the restart (MOM_driver.F90 guards save_MOM_restart with
# Restart_control>=0), and that field_checksum hits the same TIM::checksum host-
# pointer GPU bug at finalize. Negative RESTART_CONTROL disables the final restart
# entirely -> no finalize checksum. Together with READ_DEPTH_LIST=False this removes
# both checksum sites so the GPU run completes cleanly (blocker #1).
# See PROFILING_DECISIONS.md (blocker #1).
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

# File label index (mode-tagged so AMREX and FORTRAN runs don't clobber)
printf -v i0 "%03d" "$i"
tag="${KMODE,,}_${i0}"

# set_gpu_rank sets CUDA_VISIBLE_DEVICES, then execs nsys, which launches MOM6.
# --stats=true prints the CUDA kernel / memcpy / API summary tables on exit:
# they split time into device compute vs host<->device transfers vs host gaps.
# NOTE: 2> goes BEFORE the pipe so it captures MOM6's stderr (CUDA/abort
# messages), not tee's -- see the same fix in run-scaling-sweep.sh.
mpiexec -np ${NGPUS} \
    --ppn ${RANKS_PER_NODE} \
    --cpu-bind=core \
    set_gpu_rank \
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --stats=true \
        --force-overwrite=true \
        -o prof_${tag} \
        ${MOM6_EXEC} \
    2> prof_${tag}.err \
    | tee prof_${tag}.out
