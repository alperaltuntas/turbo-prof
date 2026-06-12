#!/bin/bash
# AMReX continuity profiling sweep on Derecho.
#
# Usage: sh run-profile-sweep.sh <turbo_stack> [jobsizes] [modes]
#          turbo_stack  path to the turbo-stack checkout holding the build
#                       (REQUIRED; passed straight through to run-profile.sh)
#          jobsizes  space-separated job-size indices i
#                    (default: "1 2 4 8 16 32 64 128 256 512 1024", as run-scaling-sweep.sh)
#          modes     which kernel paths to profile     (default: "FORTRAN AMREX")
#
# The analogue of run-scaling-sweep.sh, but for the *AMReX continuity* report type:
# it sweeps the problem size and, at each size, profiles both the Fortran
# (OpenMP-offload) and the AMReX continuity path under Nsight Systems. For every
# (size, mode) it delegates to run-profile.sh -- which owns the domain/timestep
# setup, the GPU-checksum workarounds (READ_DEPTH_LIST=False, RESTART_CONTROL=-1),
# and the `nsys profile --stats` invocation -- and leaves prof_<mode>_<i>.{nsys-rep,
# out,err} in the run directory for gen_amrex_report.py to parse.
#
# Run from a double_gyre run directory (MOM_input, input.nml, diag_table), on a GPU
# node. Submit job-sweep-amrex.sh (which calls this), or invoke directly in an
# interactive GPU session. The default sweeps the full 1..1024 size set (matching
# run-scaling-sweep.sh); the largest sizes make big traces and 1024 may OOM a single
# A100 (recorded as "did not complete"). Pass a shorter list to scope it down.
#
# Examples:
#   sh run-profile-sweep.sh "$TURBO_STACK"                   # default sizes, both modes
#   sh run-profile-sweep.sh "$TURBO_STACK" "16 32 64 128 256 512"
#   sh run-profile-sweep.sh "$TURBO_STACK" "32 128" AMREX    # AMReX path only, two sizes

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNPROF="${SCRIPT_DIR}/run-profile.sh"

TURBO_STACK=${1:?usage: sh run-profile-sweep.sh <turbo_stack> [jobsizes] [modes]}
JOBSIZES="${2:-1 2 4 8 16 32 64 128 256 512 1024}"
MODES="${3:-FORTRAN AMREX}"

echo "=== AMReX continuity profiling sweep @ $(date) ==="
echo "    stack: ${TURBO_STACK}"
echo "    sizes: ${JOBSIZES}"
echo "    modes: ${MODES}"
echo "    via:   ${RUNPROF}"

for i in ${JOBSIZES}; do
  for mode in ${MODES}; do
    echo "------------------------------------------------------------------"
    echo "=== profile size=${i} mode=${mode} @ $(date) ==="
    sh "${RUNPROF}" "${TURBO_STACK}" "${i}" "${mode}"
    echo "=== done size=${i} mode=${mode} rc=$? ==="
  done
done

echo "=== SWEEP COMPLETE @ $(date) ==="
