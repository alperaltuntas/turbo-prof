#!/bin/bash
#PBS -A NCGD0067
#PBS -N mom6_nsys_compare_gpu
#PBS -q main
#PBS -l walltime=02:30:00
#PBS -l select=1:ncpus=64:mpiprocs=1:ngpus=1
#PBS -l job_priority=premium

# Nsight counterpart to job-compare-gpu.sh (feeds gen_nsys_compare_report.py);
# separate so the plain sweep's wrapper stays untouched. Traced runs are slow and
# write large .nsys-rep files -- hence the long walltime; shorten JOBSIZES to scope.
#   qsub -v CONFIGS="iturbo_GPU_amrex",JOBSIZES="4 16",NRUNS=2 job-nsys-compare-gpu.sh

# nvhpc AFTER cuda so `nsys` resolves to the recorder's >=2025.5 build.
module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3

export TMPDIR=${SCRATCH}/${USER}/temp && mkdir -p $TMPDIR

CONFIGS=${CONFIGS:-"dev_turbo_GPU iturbo_GPU_amrex"}
JOBSIZES=${JOBSIZES:-"1 2 4 8 16 32 64 128 256 512"}
NRUNS=${NRUNS:-2}
RUN_DIR=${RUN_DIR:-/glade/derecho/scratch/altuntas/double_gyre.260615}

# Pass through to run-nsys-compare-sweep.sh.
[ -n "${NSTEPS}" ] && export NSTEPS
[ -n "${NSYS}" ] && export NSYS

cd ${RUN_DIR}

for config in ${CONFIGS}; do
    sh /glade/work/altuntas/turbo-prof/scripts/run-nsys-compare-sweep.sh \
        "${config}" "${JOBSIZES}" "${NRUNS}"
done
