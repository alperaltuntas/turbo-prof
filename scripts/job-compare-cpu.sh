#!/bin/bash
#PBS -A NCGD0067
#PBS -N mom6_compare_cpu
#PBS -q main
#PBS -l walltime=05:00:00
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -l job_priority=premium

# Comparison sweep, CPU configs. Override via qsub -v, e.g.:
#   qsub -v CONFIGS="dev_turbo_CPU",JOBSIZES="1 4 16",NRUNS=1 job-compare-cpu.sh

# Load modules to match compile-time environment
module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3

### Set temp to scratch
export TMPDIR=${SCRATCH}/${USER}/temp && mkdir -p $TMPDIR

CONFIGS=${CONFIGS:-"dev_turbo_CPU iturbo_CPU_amrex"}
JOBSIZES=${JOBSIZES:-"1 2 4 8 16 32 64 128 256 512 1024"}
NRUNS=${NRUNS:-3}
RUN_DIR=${RUN_DIR:-/glade/derecho/scratch/altuntas/double_gyre.260616}

cd ${RUN_DIR}

for config in ${CONFIGS}; do
    sh /glade/work/altuntas/turbo-prof/scripts/run-compare-sweep.sh \
        "${config}" "${JOBSIZES}" "${NRUNS}"
done
