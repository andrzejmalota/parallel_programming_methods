#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    const int N = 100000000;
    int i = 0;
    int *vector = malloc(N * sizeof(int));
    clock_t start, end;
    float t;
    srand(time(NULL));
    start = clock();

    for (i=0; i<N; i++) {
        vector[i] = (int) rand() % 100;
    }

    end =  clock();
    t = (float)(end - start) / CLOCKS_PER_SEC;
    printf("%f seconds \n", t);
    return 0;
}
