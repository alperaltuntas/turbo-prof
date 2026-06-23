#!/bin/bash
#PBS -A NCGD0067
#PBS -N mom6_ncu_compare_gpu
#PBS -q main
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=64:mpiprocs=1:ngpus=1
#PBS -l job_priority=premium

# Nsight Compute counterpart to job-nsys-compare-gpu.sh (drives run-ncu-compare.sh);
# separate so the nsys/plain sweeps' wrappers stay untouched. ncu REPLAYS each
# profiled kernel to collect the full counter set, so report size and record time
# scale with the NUMBER of profiled launches. The leaves are steady-state, so we
# keep the full metric set but cap profiling to the first LAUNCH_COUNT matched leaf
# launches (default 18) -- a few representative instances instead of all ~50, which
# is what keeps the .ncu-rep small (~20-30 MB) and quick to open. NSTEPS stays 10:
# once the cap is hit (early in step 1-2) the remaining steps run unprofiled at full
# speed. It profiles ONE size i per config (not a sweep). Override via qsub -v, e.g.:
#   qsub -v CONFIGS="iturbo_GPU_amrex",I=16 job-ncu-compare-gpu.sh
#   qsub -v I=64,LAUNCH_COUNT=4 job-ncu-compare-gpu.sh       # even smaller
#   qsub -v I=64,LAUNCH_SKIP=20 job-ncu-compare-gpu.sh       # skip warmup launches
#   qsub -v I=64,LAUNCH_COUNT= job-ncu-compare-gpu.sh        # uncap (legacy huge report)

# nvhpc AFTER cuda so `ncu` resolves to the recorder's build.
module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3

export TMPDIR=${SCRATCH}/${USER}/temp && mkdir -p $TMPDIR

CONFIGS=${CONFIGS:-"dev_turbo_GPU iturbo_GPU_amrex"}
I=${I:-256}
NSTEPS=${NSTEPS:-10}
RUN_DIR=${RUN_DIR:-`pwd`}

# Default the kernel filter to the three continuity PPM leaf kernels that pair
# across both builds (gen_nsys_compare_report.py); lowercase + ppm-prefix-free
# because dev_turbo emits `...ppm_reconstruction_x_<line>_gpu` and iturbo
# `...MOM::PPM_reconstruction_x(...)`. `-` (not `:-`) so NCU_KERNEL_FILTER= profiles all.
export NCU_KERNEL_FILTER=${NCU_KERNEL_FILTER-"reconstruction_x|reconstruction_y|ppm_limit_pos"}

# Cap profiling to the first X matched leaf launches (keeps the full metric set
# but a small, fast report). `-` (not `:-`) so `LAUNCH_COUNT=` explicitly uncaps.
export LAUNCH_COUNT=${LAUNCH_COUNT-18}

# Pass through to run-ncu-compare.sh.
[ -n "${NCU}" ] && export NCU
[ -n "${NCU_SET}" ] && export NCU_SET
[ -n "${LAUNCH_SKIP}" ] && export LAUNCH_SKIP
[ -n "${DUMP_CSV}" ] && export DUMP_CSV

cd ${RUN_DIR}

for config in ${CONFIGS}; do
    sh /glade/work/altuntas/turbo-prof/scripts/run-ncu-compare.sh \
        "${config}" "${I}" "${NSTEPS}"
done
