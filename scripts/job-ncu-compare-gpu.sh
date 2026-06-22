#!/bin/bash
#PBS -A NCGD0067
#PBS -N mom6_ncu_compare_gpu
#PBS -q main
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=64:mpiprocs=1:ngpus=1
#PBS -l job_priority=premium

# Nsight Compute counterpart to job-nsys-compare-gpu.sh (drives run-ncu-compare.sh);
# separate so the nsys/plain sweeps' wrappers stay untouched. ncu REPLAYS each
# kernel to collect the full counter set, so it is far slower than an nsys trace --
# hence the long walltime; it profiles ONE size i per config (not a sweep). Scope
# with a small i and the kernel/launch knobs below. Override via qsub -v, e.g.:
#   qsub -v CONFIGS="iturbo_GPU_amrex",I=16 job-ncu-compare-gpu.sh
#   qsub -v I=64,NCU_KERNEL_FILTER="regex",LAUNCH_COUNT=200 job-ncu-compare-gpu.sh

# nvhpc AFTER cuda so `ncu` resolves to the recorder's build.
module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3

export TMPDIR=${SCRATCH}/${USER}/temp && mkdir -p $TMPDIR

CONFIGS=${CONFIGS:-"dev_turbo_GPU iturbo_GPU_amrex"}
I=${I:-16}
NSTEPS=${NSTEPS:-2}
RUN_DIR=${RUN_DIR:-`pwd`}

# Pass through to run-ncu-compare.sh.
[ -n "${NCU}" ] && export NCU
[ -n "${NCU_SET}" ] && export NCU_SET
[ -n "${NCU_KERNEL_FILTER}" ] && export NCU_KERNEL_FILTER
[ -n "${LAUNCH_SKIP}" ] && export LAUNCH_SKIP
[ -n "${LAUNCH_COUNT}" ] && export LAUNCH_COUNT
[ -n "${DUMP_CSV}" ] && export DUMP_CSV

cd ${RUN_DIR}

for config in ${CONFIGS}; do
    sh /glade/work/altuntas/turbo-prof/scripts/run-ncu-compare.sh \
        "${config}" "${I}" "${NSTEPS}"
done
