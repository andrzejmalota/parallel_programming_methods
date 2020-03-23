#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

long long int get_num_pts_circle(long long int num_pts_square)
{
    const double R = 1.00;
    long long int num_pts_circle = 0, i;
    long double x, y, d;

    for (i=0; i < num_pts_square; i++)
    {
        x = (double) rand() / RAND_MAX;
        y = (double) rand() / RAND_MAX;

        d = sqrt((x * x + y * y));
        if (d <= R) { num_pts_circle++; }
    }
    return num_pts_circle;
}


int main(int argc, char ** argv) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    long long int num_pts_square = 0;

    if (argc == 2) { num_pts_square = strtoll(argv[1], NULL, 10);}
    else { exit(-1);}

    srand(time(NULL) * world_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();
    long long int num_pts_circle = get_num_pts_circle(num_pts_square / world_size);
    long long int all_points_in_circle;

    MPI_Reduce(&num_pts_circle, &all_points_in_circle, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    t = MPI_Wtime() - t;

    if (world_rank == 0)
    {
        double pi = (long double)4.0 * ((long double)all_points_in_circle / (long double)num_pts_square);
        printf("%i PROCESSORS \n", world_size);
        printf("MEAN PI: %.15f \n", pi);
        printf("TIME: %f\n\n", t);
    }
    MPI_Finalize();
    return 0;
}
