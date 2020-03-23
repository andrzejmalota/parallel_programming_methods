#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>

int main(int argc, char** argv)
{
    int rank, size;
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int len = size * 3;
    int *msg = malloc(len* sizeof(int));
    int *recv_msg = malloc(3 * sizeof(int));

    if (size < 2)
    {
        fprintf(stderr, "Size must be > 2");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0)
    {
        printf("Data before: ");
        for (i=0; i<len; i++)
        {
            msg[i] = 1;
            printf(" %d ", msg[i]);

        }
        printf("\n");
    }

    MPI_Scatter(msg, 3, MPI_INT, recv_msg, 3, MPI_INT, 0, MPI_COMM_WORLD);

    for (i = 0; i < 3; i++) {
        recv_msg[i] = recv_msg[i] * rank;
    }

    MPI_Gather(recv_msg, 3, MPI_INT, msg, 3, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Data after: ");
        for (i=0; i<len; i++)
        {

            printf(" %d ", msg[i]);
        }
        printf("\n");

    }

    free(msg);
    free(recv_msg);
    MPI_Finalize();
    return 0;
}
