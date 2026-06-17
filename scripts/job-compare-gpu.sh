#!/bin/bash
#PBS -A NCGD0067
#PBS -N mom6_compare_gpu
#PBS -q main
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=64:mpiprocs=1:ngpus=1
#PBS -l job_priority=premium

# Comparison sweep, GPU configs. Override via qsub -v, e.g.:
#   qsub -v CONFIGS="iturbo_GPU_amrex",JOBSIZES="1 4 16",NRUNS=1 job-compare-gpu.sh

# Load modules to match compile-time environment
module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3

### Set temp to scratch
export TMPDIR=${SCRATCH}/${USER}/temp && mkdir -p $TMPDIR

CONFIGS=${CONFIGS:-"dev_turbo_GPU iturbo_GPU_amrex"}
JOBSIZES=${JOBSIZES:-"1 2 4 8 16 32 64 128 256 512 1024"}
NRUNS=${NRUNS:-3}
RUN_DIR=${RUN_DIR:-/glade/derecho/scratch/altuntas/double_gyre.260616}

cd ${RUN_DIR}

for config in ${CONFIGS}; do
    sh /glade/work/altuntas/turbo-prof/scripts/run-compare-sweep.sh \
        "${config}" "${JOBSIZES}" "${NRUNS}"
done
