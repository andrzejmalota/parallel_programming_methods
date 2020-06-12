#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include "helper_functions.h"


int main() {
    int *a_d
    cudaMalloc((void **) &a_d, size); // alokuj pamięć na GPU
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    StopWatchInterface *timer=NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    // wykonaj obliczenia na GPU:
    kernel <<< n_blocks, block_size >>> (a_d, N);
    cudaThreadSynchronize();
    sdkStopTimer(&timer);
    float time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    // prześlij wyniki
    cudaMemcpy(a_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
    cudaFree(a_d);
    printf ("Time for the kernel: %f ms\n", time);
}