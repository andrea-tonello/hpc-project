#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=112
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --partition dcgp_usr_prod
#SBATCH -A uTS25_Tornator_0
#SBATCH -t 00:30:00
#SBATCH --job-name=cache_boundary
#SBATCH --output=cache_boundary.out

# Cache Boundary Test
# Fixed threads (56 = one full socket), vary problem size
# to identify the L3 cache boundary and confirm memory-bandwidth bottleneck.
#
# L3 cache per socket on DCGP = 48 MB
#  - Two planes (old + new) = 2 * N * N * 8 bytes.
#  - N = 1700 -> 46.24MB
#
# Sizes chosen to span: L2 -> L3 -> RAM

module load openmpi/4.1.6--gcc--12.2.0
mpicc -fopenmp -O2 -I include -o stencil_parallel src/stencil_template_parallel.c

EXEC=./stencil_parallel
NITER=100
NSOURCES=4
PERIODIC=0

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=56       # one full socket

OUTFILE="cache_boundary_results.csv"
echo "size,data_MB,niter,computation,ns_per_cell,GB_per_s" > ${OUTFILE}

for SIZE in 256 512 1024 1730 2048 4096 8192 16384 25000; do

    DATA_MB=$(awk "BEGIN {printf \"%.1f\", 2 * ${SIZE} * ${SIZE} * 8 / 1048576}")

    echo "Running SIZE=${SIZE}x${SIZE}  data=${DATA_MB}MB  iter=${NITER}"

    OUTPUT=$(mpirun --bind-to none -np 1 ${EXEC} \
        -x ${SIZE} -y ${SIZE} -n ${NITER} \
        -e ${NSOURCES} -p ${PERIODIC} -o 0 -O 0)

    COMP=$(echo "${OUTPUT}" | grep "Computation:" | awk '{print $2}')

    # ns per cell: time / (cells * iterations), converted to nanoseconds
    NS_PER_CELL=$(awk "BEGIN {printf \"%.4f\", \
        ${COMP} * 1e9 / (${SIZE} * ${SIZE} * ${NITER})}")

    # effective bandwidth: 6 memory ops * 8 bytes * cells * iters / time
    GB_PER_S=$(awk "BEGIN {printf \"%.2f\", \
        6 * 8 * ${SIZE} * ${SIZE} * ${NITER} / ${COMP} / 1e9}")

    echo "  Comp: ${COMP}s  ns/cell: ${NS_PER_CELL}  BW: ${GB_PER_S} GB/s"

    echo "${SIZE},${DATA_MB},${NITER},${COMP},${NS_PER_CELL},${GB_PER_S}" >> ${OUTFILE}

done

echo ""
echo "Results saved to ${OUTFILE}"
