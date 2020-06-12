// Matrix multiplication by parts
// Elements stored in row-major order

using namespace std;
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "helper_functions.h"

typedef struct
{	int width;
	int height;
	float *elements;
} Matrix;

// Forward declaration of matrix mult
__global__ void MatMulKernel (const Matrix, const Matrix, Matrix);
void MatMulCPU(const Matrix, const Matrix, Matrix);

// Host code
void MatMul(const Matrix A, const Matrix B, Matrix C, Matrix C_cpu, int blockSize)
{
    const int REPEAT_NUM = 10;
	// Load matrices A and B to device memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc((void**) &d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	
	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc((void**) &d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	
	// allocate C in device
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = d_C.width * d_C.height * sizeof(float);
	cudaMalloc((void**) &d_C.elements, size);

    StopWatchInterface *timer =NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    for (int i=0; i<REPEAT_NUM; i++)
    {
        MatMulCPU(A, B, C_cpu);
    }

    cudaThreadSynchronize();
    sdkStopTimer(&timer);
    float cpu_t = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
	
	// call kernel
    // blockSize = 32;
    int gridSize = (int)ceil((float)(d_C.width)/blockSize);

    dim3 dimBlock(blockSize, blockSize); // define the block size (what is the best value?) 
    dim3 dimGrid(gridSize, gridSize); 

    
    printf("grid size %d \n", gridSize);
    // printf("block size %d \n", blockSize);
        
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    cudaThreadSynchronize();
    sdkStopTimer(&timer);
    float gpu_t = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
	
	// copy C to host
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	
	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);

    for (int i=0; i<A.width*B.width; i++)
    {
        if (abs(C.elements[i] - C_cpu.elements[i]) > 0.0001 )
        {
            printf("GPU result not equal to CPU at index= %d \n", i);
            break;
        }
    }

    std::ofstream C_output;
    C_output.open("C_cpu.txt");
    for (int i=0; i<A.width; i++)
    {   for (int j=0; j<B.width; j++)
            C_output<<C_cpu.elements[i*A.width+j]<<"\t";
        C_output<<endl;
    }

    printf ("gpu time : %f ms, cpu time: %f ms, matrix width: %d \n", gpu_t, cpu_t/REPEAT_NUM, A.width);
}

//matrix multiplication kernel
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // each thread computes one element of C and acumulates results to Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row>=A.height) || (col>=B.width)){return;}
    for (int e=0; e<A.width; e++)
    {
        Cvalue += A.elements[row*A.width + e] * B.elements[e*B.width + col];
        C.elements[row*C.width + col] = Cvalue;
    }
}

void MatMulCPU(Matrix A, Matrix B, Matrix C)
{
    int Cvalue;
    for (int row=0; row<A.width; row++)
    {
        for (int col=0; col<A.width; col++)
        {
            Cvalue = 0;
            for (int e=0; e<A.width; e++)
            {
                Cvalue += A.elements[row*A.width + e] * B.elements[e*B.width + col];
                C.elements[row*C.width + col] = Cvalue;
            }
        }
    }
}

int main(int argc, char * const argv[])
{	
    int blockSize = atoi(argv[1]);
	int Width = atoi(argv[2]);
	
	Matrix A;
	Matrix B;
	Matrix C;
    Matrix C_cpu;
	
	A.width = Width;
	B.width = Width;
	C.width = Width;
    C_cpu.width = Width;
	
	A.height = Width;
	B.height = Width;
	C.height = Width;
    C_cpu.height = Width;
	
	A.elements = new float[Width*Width];
	B.elements = new float[Width*Width];
	C.elements = new float[Width*Width];
    C_cpu.elements = new float[Width*Width];

    // random matrix
    for (int i = 0; i < Width*Width; ++i)
        A.elements[i] = (rand() / (float)RAND_MAX) / 2;

    for (int i = 0; i < Width*Width; ++i)
        B.elements[i] = (rand() / (float)RAND_MAX) / 2;

    // matrix from file
	//fill matrices
	// std::ifstream A_input;
	// std::ifstream B_input;
	// A_input.open("A_18x18.txt");
	// B_input.open("B_18x18.txt");
	
	// float a, b;
	// A_input >> a;	
	// B_input >> b;	
	// int i = 0;
	// while (!A_input.eof())
	// {	A.elements[i] = a;
	// 	B.elements[i] = b;
	// 	A_input >> a;	
	// 	B_input >> b;	
	// 	i += 1;
	// }
	// A_input.close();
	// B_input.close();

	MatMul(A, B, C, C_cpu, blockSize);
	std::ofstream C_output;
	C_output.open("C.txt");
	for (int i=0; i<Width; i++)
	{	for (int j=0; j<Width; j++)
			C_output<<C.elements[i*Width+j]<<"\t";
		C_output<<endl;
	}

    free(A.elements);
    free(B.elements);
    free(C.elements);
    free(C_cpu.elements);
}
	
