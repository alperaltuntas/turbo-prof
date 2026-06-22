# lib-compare.sh -- shared helpers for the MOM6 double_gyre comparison runners.
#
# Sourced (not executed) by run-compare-sweep.sh, run-nsys-compare-sweep.sh and
# run-ncu-compare.sh to keep the setup that is identical across all three in one
# place: the module load, the CONFIG->STACK/exec mapping, the AMReX *_MODE env
# routing, the near-square layout helper, and the MOM_override generator. Each
# runner keeps its own TINY_ARGS, mpiexec/profiler invocation and per-run loop --
# those differ on purpose.
#
# Uses bash features (local, (( )) arithmetic); this is safe because the runners
# themselves do and are invoked through a bash-compatible `sh` on this system.
# Functions read/write the same global names the runners already use (m, n,
# STACK, PLATFORM, MOM6_EXEC, ni, nj, lx, ly, dt, dt_therm).

# Module set matching the compile-time environment. nvhpc AFTER cuda so the
# recorder's `nsys`/`ncu` (>=2025.5) resolve from nvhpc/25.9.
load_modules() {
    module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3
}

# resolve_stack <config>: set STACK, PLATFORM (cpu|gpu) and MOM6_EXEC for a
# config, then verify the executable exists. Stack roots per build variant are
# overridable via the STACK_* env vars. Exits non-zero on an unknown config or a
# missing/non-executable MOM6.
resolve_stack() {
    local config=$1
    local self
    self=$(basename "$0")

    # Default stack checkouts per build variant (override via env).
    STACK_DEV_TURBO_CPU=${STACK_DEV_TURBO_CPU:-/glade/work/altuntas/turbo-stack-dev-turbo-cpu}
    STACK_ITURBO_CPU=${STACK_ITURBO_CPU:-/glade/work/altuntas/turbo-stack-iturbo-cpu}
    STACK_DEV_TURBO=${STACK_DEV_TURBO:-/glade/work/altuntas/turbo-stack-dev-turbo}
    STACK_ITURBO=${STACK_ITURBO:-/glade/work/altuntas/turbo-stack-iturbo}

    case "${config}" in
        dev_turbo_CPU)    STACK=${STACK_DEV_TURBO_CPU} ;;
        iturbo_CPU_amrex) STACK=${STACK_ITURBO_CPU} ;;
        dev_turbo_GPU)    STACK=${STACK_DEV_TURBO} ;;
        iturbo_GPU_amrex) STACK=${STACK_ITURBO} ;;
        *) echo "${self}: unknown config: ${config}" >&2; exit 1 ;;
    esac
    case "${config}" in
        *CPU*) PLATFORM=cpu ;;
        *)     PLATFORM=gpu ;;
    esac

    MOM6_EXEC=${STACK}/bin/nvhpc/MOM6_using_TIM/MOM6/MOM6
    if [ ! -x "${MOM6_EXEC}" ]; then
        echo "${self}: MOM6 executable not found or not executable: ${MOM6_EXEC}" >&2
        echo "  check the stack for config ${config} (got: ${STACK})" >&2
        exit 1
    fi
}

# require_gpu_config <config>: for the profiling runners, which are GPU-only
# (per-kernel device timers/counters come from the GPU run). Must be called after
# resolve_stack so PLATFORM is set. Exits non-zero on a CPU config.
require_gpu_config() {
    local config=$1
    local self
    self=$(basename "$0")
    if [ "${PLATFORM}" = "cpu" ]; then
        echo "${self}: ${config} is a CPU config; profiling is" >&2
        echo "  GPU-only (per-kernel device timers come from the GPU run)." >&2
        echo "  Use run-compare-sweep.sh for the CPU configs." >&2
        exit 1
    fi
}

# set_amrex_env <config>: amrex configs route the six ported PPM kernels through
# AMReX; all others must run with these unset (the same executable serves both
# modes).
set_amrex_env() {
    local config=$1
    if [ "${config#*amrex}" != "${config}" ]; then
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
}

# Construct a square-like layout m x n for i ranks (sets globals m, n).
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

# write_mom_override <i> <nranks> <daymax>: emit MOM_override for job-size index
# <i> on <nranks> MPI ranks. Both layouts come from get_layout: the global grid
# (NIGLOBAL/NJGLOBAL/DT/DT_THERM) from <i>, and the rank decomposition (LAYOUT)
# from <nranks> -- GPU runners pass NGPUS (1 -> 1x1), the CPU sweep passes its
# weak-scaled rank count. DAYMAX is 150 for the plain sweep, NSTEPS for nsys/ncu.
write_mom_override() {
    local i=$1 nranks=$2 daymax=$3
    # `local m n` so get_layout's writes stay scoped here (bash dynamic scoping)
    # and don't leak into the caller.
    local m n
    get_layout "${nranks}"         # rank decomposition -> LAYOUT
    local lx=$m ly=$n
    get_layout "${i}"              # global grid -> NIGLOBAL/NJGLOBAL/DT
    local ni=$(( 32 * m )) nj=$(( 32 * n ))
    local dt=$(( 1200 / m )) dt_therm=$(( 2400 / m ))
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
#override DAYMAX = ${daymax}
#override READ_DEPTH_LIST = False
#override RESTART_CONTROL = -1
EOF
}
