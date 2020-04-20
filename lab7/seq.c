#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    const int N = 10000000;
    int *vector = malloc(N * sizeof(int));
    time_t t = time();
    srand(10);

    for (int i=0; i<N; i++) {
        vector[i] = (int) rand() % 100;
    }

    t = difftime(timer, time());
    printf('%.f seconds', t)
    return 0;
}
