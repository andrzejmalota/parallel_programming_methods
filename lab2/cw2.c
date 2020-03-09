#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>

int main(int argc, char** argv)
{
    int rank, size;
    int i;
    double start_t;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int len = size * 1000000;
    int *msg = malloc(len* sizeof(int));
    int *recv_msg = malloc(3 * sizeof(int));

    if (size < 2)
    {
        fprintf(stderr, "Size must be > 2");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    if (rank == 0)
    {
        printf("Data size: %d bytes", sizeof(int)*len);
//        printf("Data before: ");
        for (i=0; i<len; i++)
        {
            msg[i] = 1;
//            printf(" %d ", msg[i]);

        }
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    start_t = MPI_Wtime();
    MPI_Scatter(msg, 3, MPI_INT, recv_msg, 3, MPI_INT, 0, MPI_COMM_WORLD);
    printf("time %f", MPI_Wtime() - start_t);

    start_t = MPI_Wtime();
    MPI_Gather(recv_msg, 3, MPI_INT, msg, 3, MPI_INT, 0, MPI_COMM_WORLD);
    printf("time %f", MPI_Wtime() - start_t);


    free(msg);
    free(recv_msg);
    MPI_Finalize();
    return 0;
}
