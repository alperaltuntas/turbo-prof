#!/bin/bash
#PBS -A NCGD0067
#PBS -N mom6_standalone_gpu
#PBS -q main
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ngpus=1
#PBS -l job_priority=premium

# Load modules to match compile-time environment
module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3

### Set temp to scratch
export TMPDIR=${SCRATCH}/${USER}/temp && mkdir -p $TMPDIR

# Run dir (holds MOM_input/input.nml/diag_table) and sweep script. These two
# paths are hardcoded so the wrapper is self-contained: just `qsub` it, no
# environment forwarding needed. run-scaling-sweep.sh self-locates TURBO_STACK.
cd /glade/work/altuntas/turbo-stack-for-prof/examples/double_gyre

sh /glade/work/altuntas/turbo-prof/scripts/run-scaling-sweep.sh gpu