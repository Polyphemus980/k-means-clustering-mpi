#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function for parallel reduction
__global__ void vectorSum(int *vec, const int size)
{
    int tid = threadIdx.x;

    // Parallel reduction
    for (int stride = 1; stride < size; stride *= 2)
    {
        if (tid % (2 * stride) == 0 && (tid + stride) < size)
        {
            vec[tid] += vec[tid + stride];
        }
        __syncthreads();
    }
}

int main()
{
    const int SIZE = 5;

    // Host data
    int h_vec[SIZE] = {1, 2, 3, 4, 5}; // Input vector
    int result = 0;

    // Device data
    int *d_vec = nullptr;

    // Allocate memory on device
    cudaMalloc(&d_vec, SIZE * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_vec, h_vec, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 5 threads in one block
    vectorSum<<<1, SIZE>>>(d_vec, SIZE);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(&result, d_vec, sizeof(int), cudaMemcpyDeviceToHost);

    // Print original vector
    printf("Original vector: ");
    for (int i = 0; i < SIZE; i++)
    {
        printf("%d ", h_vec[i]);
    }
    printf("\n");

    // Print result
    printf("Sum: %d\n", result); // Should print 15 (1+2+3+4+5)

    // Calculate CPU result for verification
    int cpu_sum = 0;
    for (int i = 0; i < SIZE; i++)
    {
        cpu_sum += h_vec[i];
    }
    printf("CPU Sum: %d\n", cpu_sum);

    // Clean up
    cudaFree(d_vec);

    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    return 0;
}