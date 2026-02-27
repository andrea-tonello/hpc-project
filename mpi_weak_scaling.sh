#!/bin/bash

#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --partition dcgp_usr_prod
#SBATCH -A uTS25_Tornator_0
#SBATCH -t 01:00:00
#SBATCH --job-name=mpi_weak
#SBATCH --output=mpi_weak_scaling.out

# MPI Weak Scaling
# Loop uses 1,2,4,8,16 nodes via rankfile.
# YSIZE scales linearly with nodes (1D Y decomposition): work per task stays constant.
# Each node: 8 MPI tasks * 14 OpenMP threads = 112 cores.

module load openmpi/4.1.6--gcc--12.2.0
mpicc -fopenmp -O3 -march=native -I include -o stencil_parallel src/stencil_template_parallel.c

EXEC=./stencil_parallel

XSIZE=16384
YSIZE_PER_NODE=16384   # Each node always processes this many rows (fixed workload per node)
NITER=100
NSOURCES=25
PERIODIC=0

TASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}
THREADS_PER_TASK=${SLURM_CPUS_PER_TASK}

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=${THREADS_PER_TASK}

# Expand nodelist once; index into it per iteration
HOSTNAMES=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))

OUTFILE="mpi_weak_scaling_results.csv"
echo "nodes,tasks,ysize,elapsed,computation,communication" > ${OUTFILE}

for NODES in 1 2 4 8 16; do

    TOTAL_TASKS=$(( NODES * TASKS_PER_NODE ))
    TOTAL_YSIZE=$(( YSIZE_PER_NODE * NODES ))

    # Build rankfile for the first $NODES nodes only
    RANK_FILE=rank_file_weak_${NODES}
    rm -f ${RANK_FILE}
    rank=0
    for ((n=0; n<NODES; n++)); do
        node=${HOSTNAMES[$n]}
        for ((r=0; r<TASKS_PER_NODE; r++)); do
            start_core=$(( r * THREADS_PER_TASK ))
            end_core=$(( start_core + THREADS_PER_TASK - 1 ))
            echo "rank $rank=$node slot=$start_core-$end_core" >> ${RANK_FILE}
            ((rank++))
        done
    done

    echo "Running with ${NODES} nodes (${TOTAL_TASKS} MPI tasks, grid ${XSIZE}x${TOTAL_YSIZE})"

    OUTPUT=$(mpirun --rankfile ${RANK_FILE} ${EXEC} \
        -x ${XSIZE} -y ${TOTAL_YSIZE} -n ${NITER} \
        -e ${NSOURCES} -p ${PERIODIC} -o 0 -O 0)

    ELAPSED=$(echo "${OUTPUT}" | grep "Elapsed time:" | awk '{print $3}')
    COMP=$(echo "${OUTPUT}"   | grep "Computation:"  | awk '{print $2}')
    COMM=$(echo "${OUTPUT}"   | grep "Communication:" | awk '{print $2}')

    echo "  Elapsed: ${ELAPSED}s  Comp: ${COMP}s  Comm: ${COMM}s"
    echo "${NODES},${TOTAL_TASKS},${TOTAL_YSIZE},${ELAPSED},${COMP},${COMM}" >> ${OUTFILE}

    rm -f ${RANK_FILE}

done

echo ""
echo "Results saved to ${OUTFILE}"
