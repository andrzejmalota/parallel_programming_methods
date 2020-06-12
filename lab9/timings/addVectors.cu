#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include "helper_functions.h"


__global__ void add (int *a, int *b, int *c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N) {
        c[tid] = a[tid]+b[tid];
    }
}

void cpu_add(int *a, int *b, int *c, int N)
{
    int i;
    for(i=0; i<N; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]) {
    long int N = atoi(argv[1]);
    int blockSize = atoi(argv[2]);

    int *dev_a, *dev_b, *dev_c;
    cudaError_t err = cudaSuccess;


    int *a = (int*)malloc(N * sizeof(int));
    int *b = (int*)malloc(N * sizeof(int));
    int *c = (int*)malloc(N * sizeof(int));
    int *c_cpu = (int*)malloc(N * sizeof(int));

    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i*2;
    }



    err = cudaMalloc((void**)&dev_a,N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&dev_b,N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&dev_c,N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice);

    int gridSize = (N + blockSize - 1) / blockSize;
    if(gridSize > 64) gridSize = 64;

    StopWatchInterface *timer =NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    add <<<gridSize, blockSize>>> (dev_a,dev_b,dev_c, N);

    cudaThreadSynchronize();
    sdkStopTimer(&timer);
    float gpu_t = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);

    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);


    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    cpu_add(a, b, c_cpu, N);

    sdkStopTimer(&timer);
    float cpu_t = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);

    bool equal = 1;

    for (long int i=0; i<N; i++) {
        if (fabs(c[i] - c_cpu[i] > 1e-5)) equal = 0;
    }

    if (equal) {
        printf("Cpu result is equal to gpu \n");
    }
    else {
        printf("Cpu result is not equal to gpu \n");
    }

    printf ("Vector size: %ld bytes, gpu time : %f ms, cpu time: %f ms, block size: %d, grid size: %d \n",
     N * sizeof(int), gpu_t, cpu_t, blockSize, gridSize);


    return 0;
}

