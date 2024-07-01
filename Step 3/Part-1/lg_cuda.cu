#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "nvToolsExt.h"


// CUDA kernel for computing cost
__global__ void compute_cost_kernel(const float* h, const float* y, float* cost, int m) {
    __shared__ float temp[256];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float loss = 0.0;
    
    if (idx < m) {
        loss = -y[idx] * logf(h[idx]) - (1 - y[idx]) * logf(1 - h[idx]);
    }
    
    temp[threadIdx.x] = loss;
    __syncthreads();
    
    // Reduction to sum up the loss
    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            temp[threadIdx.x] += temp[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(cost, temp[0] / m);
    }
}

// Sigmoid function called by other device functions
__device__ float sigmoid(float z) {
    return 1.0 / (1.0 + expf(-z));
}

// compute new hpothesis
__global__ void compute_hypothesis(const float* X, const float* theta, float* h, int n_samples, int n_features) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_samples) {
        float z = 0.0;
        for (int j = 0; j < n_features; j++) {
            z += X[idx * n_features + j] * theta[j];
        }
        h[idx] = sigmoid(z);
    }
}

// compute new gradient
__global__ void compute_gradient(const float* X, const float* y, const float* h, float* gradient, int n_samples, int n_features) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_features) {
        float grad = 0.0;
        for (int i = 0; i < n_samples; i++) {
            grad += (y[i] - h[i]) * X[i * n_features + idx];
        }
        gradient[idx] = grad / n_samples;
    }
}

// Update weights
__global__ void update_weights(float* theta, const float* gradient, float learning_rate, int n_features) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_features) {
        theta[idx] += learning_rate * gradient[idx];
    }
}

int main() {
    cudaSetDevice(6);
    
    const int n_samples = 10000;
    const int n_features = 3;
    const float w_true[] = {-1.0, 1.2, 1.0};
    const float sigma = 0.1;
    const float learning_rate = 0.01;
    const int num_iterations = 100;

    float *X, *y, *theta, *h, *gradient, *cost;
    float *d_X, *d_y, *d_theta, *d_h, *d_gradient, *d_cost;

    // Allocate some cpu memory
    X = (float*)malloc(n_samples * n_features * sizeof(float));
    y = (float*)malloc(n_samples * sizeof(float));
    theta = (float*)malloc(n_features * sizeof(float));
    h = (float*)malloc(n_samples * sizeof(float));
    gradient = (float*)malloc(n_features * sizeof(float));
    cost = (float*)malloc(sizeof(float));

    // Initialize host memory
    srand(0);
    for (int i = 0; i < n_samples; i++) {
        X[i * n_features] = 1.0; // Intercepts
        for (int j = 1; j < n_features; j++) {
            X[i * n_features + j] = ((float)rand() / RAND_MAX) + sigma * ((float)rand() / RAND_MAX);
        }
        y[i] = w_true[0] + w_true[1] * X[i * n_features + 1] + w_true[2] * X[i * n_features + 2] >= 0 ? 1.0 : 0.0;
    }
    for (int j = 0; j < n_features; j++) {
        theta[j] = 0.0;
    }

    // Allocate device memory
    cudaMalloc(&d_X, n_samples * n_features * sizeof(float));
    cudaMalloc(&d_y, n_samples * sizeof(float));
    cudaMalloc(&d_theta, n_features * sizeof(float));
    cudaMalloc(&d_h, n_samples * sizeof(float));
    cudaMalloc(&d_gradient, n_features * sizeof(float));
    cudaMalloc(&d_cost, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_X, X, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, theta, n_features * sizeof(float), cudaMemcpyHostToDevice);

    int block_dim = 64;
    int grid_dim_samples = (n_samples + block_dim - 1) / block_dim;
    int grid_dim_features = (n_features + block_dim - 1) / block_dim;

    for (int iter = 0; iter < num_iterations; iter++) {

        nvtxRangePushA("Compute Hypothesis");
        compute_hypothesis<<<grid_dim_samples, block_dim>>>(d_X, d_theta, d_h, n_samples, n_features);
        nvtxRangePop();

        nvtxRangePushA("Compute Gradient");
        compute_gradient<<<grid_dim_features, block_dim>>>(d_X, d_y, d_h, d_gradient, n_samples, n_features);
        nvtxRangePop();

        nvtxRangePushA("Update Weights");
        update_weights<<<grid_dim_features, block_dim>>>(d_theta, d_gradient, learning_rate, n_features);
        nvtxRangePop();

        nvtxRangePushA("Compute Cost");
        cudaMemset(d_cost, 0, sizeof(float));
        compute_cost_kernel<<<grid_dim_samples, block_dim>>>(d_h, d_y, d_cost, n_samples);
        cudaMemcpy(cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost);
        nvtxRangePop();
        // printf("Iteration %d: Cost = %f\n", iter + 1, *cost);
    }

    // Copy results back to host
    cudaMemcpy(theta, d_theta, n_features * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Theta after gradient descent:\n");
    for (int j = 0; j < n_features; j++) {
        printf("%f ", theta[j]);
    }
    printf("\n");

    // Free memory
    free(X);
    free(y);
    free(theta);
    free(h);
    free(gradient);
    free(cost);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_theta);
    cudaFree(d_h);
    cudaFree(d_gradient);
    cudaFree(d_cost);

    return 0;
}
