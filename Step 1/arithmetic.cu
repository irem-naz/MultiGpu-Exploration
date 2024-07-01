#include <iostream>
#include <cuda_runtime.h>

__global__ void simpleKernel(int *d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_data[idx] = idx;
    }
}

int main(int argc, char *argv[]) {
    int device = 3;
    cudaSetDevice(device);

    int size = 1024;
    int *h_data = new int[size];
    int *d_data;

    cudaMalloc(&d_data, size * sizeof(int));

    // Configure the block and grid size
    int threadsPerBlock = 1024;
    int blocksPerGrid = size / threadsPerBlock;

    simpleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    delete[] h_data;

    cudaDeviceReset();

    std::cout << "CUDA program completed successfully." << std::endl;
    return 0;
}
