#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nvml.h>
#include <vector>
#include <chrono>


int GPU_N = 1;

void read_mnist_data(const char* filename, float** X, int** y, int* n_samples, int* n_features, int fraction) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read the shape of X
    int X_shape[2];
    fread(X_shape, sizeof(int), 2, file);
    int original_n_samples = X_shape[0];
    *n_features = X_shape[1];
    *n_samples = original_n_samples * fraction;  // Multiply number of samples by fraction

    // Allocate memory for X
    *X = (float*)malloc((*n_samples) * (*n_features) * sizeof(float));
    if (*X == NULL) {
        fprintf(stderr, "Failed to allocate host memory for X\n");
        exit(EXIT_FAILURE);
    }

    // Read the original data
    float* original_X = (float*)malloc(original_n_samples * (*n_features) * sizeof(float));
    fread(original_X, sizeof(float), original_n_samples * (*n_features), file);

    // Repeat the data
    for (int i = 0; i < fraction; i++) {
        memcpy(*X + i * original_n_samples * (*n_features), original_X, original_n_samples * (*n_features) * sizeof(float));
    }

    // Read and repeat the labels in the same way
    int y_shape[1];
    fread(y_shape, sizeof(int), 1, file);
    *y = (int*)malloc((*n_samples) * sizeof(int));
    if (*y == NULL) {
        fprintf(stderr, "Failed to allocate host memory for y\n");
        exit(EXIT_FAILURE);
    }

    int* original_y = (int*)malloc(original_n_samples * sizeof(int));
    fread(original_y, sizeof(int), original_n_samples, file);

    for (int i = 0; i < fraction; i++) {
        memcpy(*y + i * original_n_samples, original_y, original_n_samples * sizeof(int));
    }
    free(original_X);
    free(original_y);
    fclose(file);
}


__global__ void euclidean_distance_kernel(float* X_train, float* X_test, float* distances, int train_size, int test_size, int feature_size) {
    int train_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int test_idx = blockIdx.y;

    if (train_idx < train_size && test_idx < test_size) {
        float dist = 0.0;
        for (int i = 0; i < feature_size; i++) {
            float train_val = X_train[train_idx * feature_size + i];
            float test_val = X_test[test_idx * feature_size + i];
            float diff = train_val - test_val;
            dist += diff * diff;
        }
        distances[test_idx * train_size + train_idx] = sqrtf(dist);
    }
}

__global__ void weighted_voting_kernel(float* distances, int* y_train, int* predictions, int test_size, int train_size, int k, int max_label, float* neighbors_distances, int* neighbors_labels) {
    int test_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (test_idx < test_size) {
        float* local_neighbors_distances = &neighbors_distances[test_idx * k];
        int* local_neighbors_labels = &neighbors_labels[test_idx * k];

        for (int i = 0; i < k; i++) {
            local_neighbors_distances[i] = 1e10;
            local_neighbors_labels[i] = -1;
        }

        for (int i = 0; i < train_size; i++) {
            float dist = distances[test_idx * train_size + i];
            for (int j = 0; j < k; j++) {
                if (dist < local_neighbors_distances[j]) {
                    for (int l = k - 1; l > j; l--) {
                        local_neighbors_distances[l] = local_neighbors_distances[l - 1];
                        local_neighbors_labels[l] = local_neighbors_labels[l - 1];
                    }
                    local_neighbors_distances[j] = dist;
                    local_neighbors_labels[j] = y_train[i];
                    break;
                }
            }
        }

        float weighted_votes[10];
        for (int i = 0; i < max_label; i++) {
            weighted_votes[i] = 0.0;
        }

        for (int i = 0; i < k; i++) {
            weighted_votes[local_neighbors_labels[i]] += 1.0 / (local_neighbors_distances[i] + 1e-5);
        }

        float max_vote = 0.0;
        int pred_label = -1;
        for (int i = 0; i < max_label; i++) {
            if (weighted_votes[i] > max_vote) {
                max_vote = weighted_votes[i];
                pred_label = i;
            }
        }
        predictions[test_idx] = pred_label; 
    }
}

// void report_global_mem() {
//     nvmlInit();
//     for (int i = 0; i < GPU_N; i++) {
//         nvmlDevice_t handle;
//         nvmlDeviceGetHandleByIndex(i, &handle);
//         nvmlMemory_t mem_info;
//         nvmlDeviceGetMemoryInfo(handle, &mem_info);
//         printf("GPU %d:\n", i);
//         printf("  Total memory: %.2f MB\n", mem_info.total / (1024.0 * 1024.0));
//         printf("  Free memory: %.2f MB\n", mem_info.free / (1024.0 * 1024.0));
//         printf("  Used memory: %.2f MB\n", mem_info.used / (1024.0 * 1024.0));
//     }
//     nvmlShutdown();
// }

int main(int argc, char* argv[]) {
    printf("Running with %i GPUs...\n", GPU_N);

    // Fetch and prepare data
    const char* filename = "mnist_data.bin";
    float* X;
    int* y;
    int n_samples, n_features;

    int fraction = 1;  // Default fraction
    if (argc > 1) {
        GPU_N = atoi(argv[1]);  // Read fraction from command line
    }

    if (argc > 2) {
        fraction = atoi(argv[2]);
    }

    // Read data from file
    read_mnist_data(filename, &X, &y, &n_samples, &n_features, fraction);

    // Split data into train and test sets
    int train_size = (int)(n_samples * 0.75);
    int test_size = n_samples - train_size;
    int n_classes = 10;
    float *X_train = X;
    float *X_test = X + train_size * n_features;
    int *y_train = y;
    int *y_test = y + train_size;

    // report_global_mem();

    auto start_time = std::chrono::high_resolution_clock::now();

    // Allocate device memory
    float *d_Xtrain[GPU_N], *d_Xtest[GPU_N], *d_distances[GPU_N];
    int *d_yTrain[GPU_N], *d_predictions[GPU_N];
    for (int i = 0; i < GPU_N; i++) {
        cudaSetDevice(i);
        cudaMalloc(&d_Xtrain[i], train_size * n_features * sizeof(float));
        cudaMalloc(&d_Xtest[i], test_size / GPU_N * n_features * sizeof(float));
        cudaMalloc(&d_yTrain[i], train_size * sizeof(int));
        cudaMalloc(&d_distances[i], test_size / GPU_N * train_size * sizeof(float));
        cudaMalloc(&d_predictions[i], test_size / GPU_N * sizeof(int));
    }


    

    // Copy data to device
    for (int i = 0; i < GPU_N; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(d_Xtrain[i], X_train, train_size * n_features * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_Xtest[i], X_test + i * (test_size / GPU_N) * n_features, test_size / GPU_N * n_features * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_yTrain[i], y_train, train_size * sizeof(int), cudaMemcpyHostToDevice);
    }


    
    // Launch distance calculation kernel
    int threads_per_block = 256;
    dim3 blocks_per_grid((train_size + threads_per_block - 1) / threads_per_block, test_size / GPU_N);
    for (int i = 0; i < GPU_N; i++) {
        cudaSetDevice(i);
        euclidean_distance_kernel<<<blocks_per_grid, threads_per_block>>>(d_Xtrain[i], d_Xtest[i], d_distances[i], train_size, test_size / GPU_N, n_features);
    }
    

    // Launch weighted voting kernel
    int k = 3;
    int max_label = n_classes;
    float *neighbors_distances[GPU_N];
    int *neighbors_labels[GPU_N];
    for (int i = 0; i < GPU_N; i++) {
        cudaSetDevice(i);
        cudaMalloc(&neighbors_distances[i], test_size / GPU_N * k * sizeof(float));
        cudaMalloc(&neighbors_labels[i], test_size / GPU_N * k * sizeof(int));
        weighted_voting_kernel<<<(test_size / GPU_N + threads_per_block - 1) / threads_per_block, threads_per_block>>>(d_distances[i], d_yTrain[i], d_predictions[i], test_size / GPU_N, train_size, k, max_label, neighbors_distances[i], neighbors_labels[i]);
    }

    // Copy predictions back to host
    int *h_predictions = (int*) malloc(test_size * sizeof(int));
    for (int i = 0; i < GPU_N; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(h_predictions + i * (test_size / GPU_N), d_predictions[i], test_size / GPU_N * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Evaluate accuracy
    int correct = 0;
    for (int i = 0; i < test_size; i++) {
        if (h_predictions[i] == y_test[i]) {
            correct++;
        }
    }
    printf("Accuracy: %.2f%%\n", 100.0 * correct / test_size);
    
    // Cleanup
    free(X);
    free(y);
    free(h_predictions);
    for (int i = 0; i < GPU_N; i++) {
        cudaSetDevice(i);
        cudaFree(d_Xtrain[i]);
        cudaFree(d_Xtest[i]);
        cudaFree(d_yTrain[i]);
        cudaFree(d_distances[i]);
        cudaFree(d_predictions[i]);
        cudaFree(neighbors_distances[i]);
        cudaFree(neighbors_labels[i]);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    printf("Elapsed time: %f s\n", elapsed.count());


    return 0;
}
