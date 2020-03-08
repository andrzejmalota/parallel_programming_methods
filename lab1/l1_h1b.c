#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>

int main(int argc, char** argv)
{
	int rank, size;
	const int N = 10000;	
	const int MSG_SIZE = 1048576;
	int msg[MSG_SIZE];
	int i;
	double  avg_delay, avg_bandwidth;
	double start_t, elapsed_t;	

	MPI_Status stat;
	MPI_Request req1, req2;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size < 2) 
	{
		fprintf(stderr, "Size must be > 2");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0)
	{
		start_t = MPI_Wtime();
		for (i=0; i<N; i++)
		{
			MPI_Isend(msg, MSG_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, &req1);
			MPI_Irecv(msg, MSG_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, &req2);
			MPI_Wait(&req1, &stat);	
	}
		elapsed_t = MPI_Wtime() - start_t;
		avg_delay = elapsed_t / (2 * N);
		avg_bandwidth = sizeof(msg) / avg_delay * 8 / (1024*1000);
		printf("Size of msg %d bits \n", (int)(sizeof(msg)*8));
		printf("Avg msg delay: %g s \n", avg_delay);
		printf("Avg bandwidth: %g Mbit/s \n", avg_bandwidth);
	}
	
	if (rank == 1)
	{
		for (i=0; i<N; i++)
		{
			MPI_Isend(msg, MSG_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, &req1);
			MPI_Irecv(msg, MSG_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, &req2);
			MPI_Wait(&req2, &stat);
		}
	}
	//free(msg);	
	MPI_Finalize();
	return 0;
}
		
		
 


















