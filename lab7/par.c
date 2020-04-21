#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int num_threads = atoi(argv[1]);   
    omp_set_num_threads(num_threads);
    const int N = 100000000;
    int i = 0, j = 0;
    int *vector = malloc(N * sizeof(int));
    float start, end, t;
    start = omp_get_wtime();
    unsigned short xi[num_threads];
    #pragma omp parallel shared (vector, start, end) private(xi, i)
    {
    for (j=0; j<num_threads; j++) {
	xi[j] = j;
    } 
    #pragma omp for schedule(guided, 8)
    for (i=0; i<N; i++) {
        vector[i] = (int) erand48(xi) * 10000;
    }
    }

    end =  omp_get_wtime();
    t = end - start;
    printf("%g seconds \n", t);
    free(vector);
    return 0;
}
