#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=112
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --partition dcgp_usr_prod
#SBATCH -A uTS25_Tornator_0
#SBATCH -t 00:30:00
#SBATCH --job-name=omp_scaling

# ============================================================
# OpenMP Scaling Study
# 1 MPI task, vary OMP_NUM_THREADS: 1,2,4,8,16,32,56,84,112
# Fixed problem size (strong scaling within a single node)
# ============================================================

mpicc -fopenmp -O2 -I include -o stencil_parallel src/stencil_template_parallel.c

EXEC=./stencil_parallel

# Problem parameters - large enough so 1-thread run isn't too long
# but small enough to fit in memory on 1 node.
# Adjust -n (iterations) to control total runtime.
XSIZE=5000
YSIZE=5000
NITER=500
NSOURCES=25
PERIODIC=0

OUTFILE="omp_scaling_results.csv"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Header
echo "threads,elapsed,computation,communication" > ${OUTFILE}

for THREADS in 1 2 4; do

    export OMP_NUM_THREADS=${THREADS}

    echo "Running with OMP_NUM_THREADS=${THREADS}"

    # Run and capture output
    OUTPUT=$(mpirun -np 1 --bind-to none ${EXEC} \
        -x ${XSIZE} -y ${YSIZE} -n ${NITER} \
        -e ${NSOURCES} -p ${PERIODIC} -o 1 -O 0)

    # Extract timing values
    ELAPSED=$(echo "${OUTPUT}" | grep "Elapsed time:" | awk '{print $3}')
    COMP=$(echo "${OUTPUT}" | grep "Computation:" | awk '{print $2}')
    COMM=$(echo "${OUTPUT}" | grep "Communication:" | awk '{print $2}')

    echo "  Elapsed: ${ELAPSED}s  Comp: ${COMP}s  Comm: ${COMM}s"

    # Append to CSV
    echo "${THREADS},${ELAPSED},${COMP},${COMM}" >> ${OUTFILE}

done

echo ""
echo "Results saved to ${OUTFILE}"
