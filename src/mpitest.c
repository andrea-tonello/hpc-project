#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int Rank, Ntasks;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Ntasks);

    if (Ntasks != 2) {
        if (Rank == 0)
            printf("This test requires exactly 2 MPI tasks. Run with: mpirun -np 2 ./mpitest\n");
        MPI_Finalize();
        return 1;
    }

    double value;

    if (Rank == 0) {
        value = 42.0;
        printf("Rank 0: sending value %.1f to rank 1\n", value);
        MPI_Send(&value, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&value, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank 0: received value %.1f from rank 1\n", value);
    }

    if (Rank == 1) {
        MPI_Recv(&value, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank 1: received value %.1f from rank 0\n", value);
        value = 99.0;
        printf("Rank 1: sending value %.1f to rank 0\n", value);
        MPI_Send(&value, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
