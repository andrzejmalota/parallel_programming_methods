#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int num_threads = atoi(argv[1]);   
    omp_set_num_threads(num_threads);
    const int N = 200000000;
    const int NUM_RUNS = 10;
    int i = 0, j = 0;
    long long int chunk_size;
    int *vector = malloc(N * sizeof(int));
    float start, end, t;

    for (chunk_size=1000; chunk_size<100000000; chunk_size*=100) {
    start = omp_get_wtime();
    for (j=0; j<NUM_RUNS; j++) {
    #pragma omp parallel shared (vector, start, end, chunk_size) private(i)
    {
    int id = omp_get_thread_num();
    srand(time(0)*(1+id)); 
    #pragma omp for
    for (i=0; i<N; i++) {
        vector[i] = (int) rand_r(&id) * 10000;
    }
    }
    }
    end =  omp_get_wtime();
    t = end - start;

    printf("Chunk size: %d, %g s \n", chunk_size, t/NUM_RUNS);
    
    }
    free(vector);
    return 0;
}
