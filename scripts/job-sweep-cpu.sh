#!/bin/bash
#PBS -A NCGD0067
#PBS -N mom6_standalone_cpu
#PBS -q main
#PBS -l walltime=02:00:00
# Derecho CPU nodes are dual-socket: 128 cores. The sweep packs one node and
# caps ranks at CPU_PER_NODE=128, so one 128-core node serves every job size.
#PBS -l select=1:ncpus=128:mpiprocs=128

# Load modules to match compile-time environment
module load ncarenv/25.10 cuda/12.9.0 hdf5/1.14.6 nvhpc/25.9 ncarcompilers/1.1.0 netcdf/4.9.3

### Set temp to scratch
export TMPDIR=${SCRATCH}/${USER}/temp && mkdir -p $TMPDIR

# Run dir (holds MOM_input/input.nml/diag_table) and sweep script. These two
# paths are hardcoded so the wrapper is self-contained: just `qsub` it, no
# environment forwarding needed. run-scaling-sweep.sh self-locates TURBO_STACK.
cd /glade/work/altuntas/turbo-stack-for-prof/examples/double_gyre

sh /glade/work/altuntas/turbo-prof/scripts/run-scaling-sweep.sh cpu
