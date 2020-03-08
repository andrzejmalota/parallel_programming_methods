#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int bufsize = MPI_BSEND_OVERHEAD;
  char *buf = (char *)malloc( bufsize );
  char msg[10] = "hi!";
  char recv_msg[10];

  // We are assuming at least 2 processes for this task
  if (world_size < 2) {
    fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1); 
  }
  if (world_rank == 0) {
    // If we are rank 0, set the number to -1 and send it to process 1
    MPI_Buffer_attach(buf, bufsize);
    MPI_Rsend(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    MPI_Recv(&recv_msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Process 1 received message %s from process 0\n", recv_msg);
  }
  MPI_Finalize();
  return 0;
}
