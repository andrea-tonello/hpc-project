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

# Cache boundary test
# Fixed threads (56, all on socket 0 / CPUs 0-55), vary problem size
# to identify the L3 cache boundary and confirm memory-bandwidth bottleneck.
#
# L3 cache per socket on DCGP: 105 MB
#  - Two planes: OLD + NEW = 2 * N * N * 8 bytes
#  - But the stencil reads from OLD and writes to NEW. The heavy work occurs with the OLD plane:
#     - old[i,j] -> read by: (i,j), (i+1,j), (i-1,j), (i,j+1), (i,j-1) -> keep in cache
#     - new[i,j] -> written once, next read is SIZE^2 elements later -> discard freely
#
#  -> Formula becomes N * N * 8 bytes
#  - An interesting N would be 3500 -> 3500 * 3500 * 8 = 98MB < 105MB 
#  - N = 4096 -> 134MB, out of L3 -> We should see a decrease in bandwidth (spills into RAM, capped at 614 GB/s)


module load openmpi/4.1.6--gcc--12.2.0
mpicc -fopenmp -O3 -march=native -I include -o stencil_parallel src/stencil_template_parallel.c

EXEC=./stencil_parallel
NITER=100
NSOURCES=4
PERIODIC=0

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=56       # one full socket

OUTFILE="cache_boundary_results.csv"
echo "size,data_MB,comp_time,GB_per_s" > ${OUTFILE}

for SIZE in 512 1024 2048 3500 4096 8192 16384 32768; do

    DATA_MB=$(awk "BEGIN {printf \"%.1f\", ${SIZE} * ${SIZE} * 8 / 1e6}")

    echo "Running SIZE=${SIZE}x${SIZE}  data=${DATA_MB}MB  iter=${NITER}"

    OUTPUT=$(mpirun --bind-to none -np 1 ${EXEC} \
        -x ${SIZE} -y ${SIZE} -n ${NITER} \
        -e ${NSOURCES} -p ${PERIODIC} -o 0 -O 0)

    COMP=$(echo "${OUTPUT}" | grep "Computation:" | awk '{print $2}')

    # effective bandwidth: 6 memory ops * 8 bytes * cells * iters / time
    GB_PER_S=$(awk "BEGIN {printf \"%.2f\", \
        6 * 8 * ${SIZE} * ${SIZE} * ${NITER} / ${COMP} / 1e9}")

    echo "  Comp: ${COMP}s  ns/cell: ${NS_PER_CELL}  BW: ${GB_PER_S} GB/s"

    echo "${SIZE},${DATA_MB},${COMP},${GB_PER_S}" >> ${OUTFILE}

done

echo ""
echo "Results saved to ${OUTFILE}"
