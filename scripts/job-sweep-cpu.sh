#!/bin/bash
#PBS -A NCGD0067
#PBS -N mom6_standalone_cpu
#PBS -q main
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -l job_priority=premium

# Load modules to match compile-time environment
module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3

### Set temp to scratch
export TMPDIR=${SCRATCH}/${USER}/temp && mkdir -p $TMPDIR

TURBO_STACK=/glade/work/altuntas/turbo-stack-dev-turbo-cpu

cd ${TURBO_STACK}/examples/double_gyre

sh /glade/work/altuntas/turbo-prof/scripts/run-scaling-sweep.sh "${TURBO_STACK}" cpu
