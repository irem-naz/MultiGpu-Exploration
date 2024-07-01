from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import cupy as cp
import numpy as np
import time
import nvtx

# self-made train_test_split alogrithm less complex than the sklearn's
def train_test_split_gpu(X, y, test_size=0.25, random_state=None):
    if random_state is not None:
        cp.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = cp.arange(n_samples)
    cp.random.shuffle(indices)
    
    test_size = int(n_samples * test_size)
    train_size = n_samples - test_size
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    # print(isinstance(X, cp.ndarray))
    # print(isinstance(X_train, cp.ndarray))
    
    return X_train, X_test, y_train, y_test

# Using train_test_split algorithm to only get the indices using cpu
def get_train_test_indices(n_samples, test_size=0.25, random_state=None):
    indices = np.arange(n_samples)
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    return train_indices, test_indices
# Using the indices produced from CPU/train_test_split to index the X and y arrays in GPU storage
def get_split_arrays(X, y, train_indices, test_indices):
    # Convert indices to CuPy arrays
    train_indices_gpu = cp.asarray(train_indices)
    test_indices_gpu = cp.asarray(test_indices)

    # Split the data using the indices
    X_train = X[train_indices_gpu]
    X_test = X[test_indices_gpu]
    y_train = y[train_indices_gpu]
    y_test = y[test_indices_gpu]

    return X_train, X_test, y_train, y_test


# using a k nearest neighbour algorithm since it is compute intensive while measuring 
# distance for each test value to every train value for classification, it can also be scaled
# by k value (nearest 3, 4, 5 etc neighbours)

# CUDA kernel for computing Euclidean distance

# the train, test value are not passed?? printing only 0s!! hence the distances give 0 only as well
euclidean_distance_kernel = cp.RawKernel(r'''
extern "C" __global__
void euclidean_distance_kernel(float* X_train, float* X_test, float* distances, int train_size, int feature_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < train_size) {
        float dist = 0.0;
        for (int i = 0; i < feature_size; i++) {
            //float diff = X_train[tid * feature_size + i] - X_test[i];
            //dist += diff * diff;
            float train_val = X_train[tid * feature_size + i];
            float test_val = X_test[i];
            float diff = train_val - test_val;
            dist += diff * diff;
            //printf("tid: %d, i: %d, train_val: %f, test_val: %f, diff: %f, dist: %f\n", 
                    //tid, i, train_val, test_val, diff, dist);
        }
        distances[tid] = sqrtf(dist);
        //printf("tid: %d, dist: %f\n", tid, distances[tid]);
    }
}
''', 'euclidean_distance_kernel')

# CUDA kernel for predicting labels using k-nearest neighbors
predict_kernel = cp.RawKernel(r'''
extern "C" __global__
void predict_kernel(int* y_train, float* distances, int* indices, int* y_pred, int n, int test_size, int train_size, int max_label) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < test_size) {
        // Sort distances and get k-nearest neighbors
        for (int i = 0; i < train_size; i++) {
            for (int j = i + 1; j < train_size; j++) {
                if (distances[i] > distances[j]) {
                    float temp_dist = distances[i];
                    distances[i] = distances[j];
                    distances[j] = temp_dist;
                    int temp_idx = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp_idx;
                }
            }
        }

        int label_counts[10];
        for (int i = 0; i <= max_label; i++) {
            label_counts[i] = 0;
        }

        for (int i = 0; i < n; i++) {
            label_counts[y_train[indices[i]]]++;
        }

        int max_count = 0;
        int predicted_label = 0;
        for (int i = 0; i <= max_label; i++) {
            if (label_counts[i] > max_count) {
                max_count = label_counts[i];
                predicted_label = i;
            }
        }

        y_pred[tid] = predicted_label;
    }
}
''', 'predict_kernel')

# Compute Euclidean distance using the CUDA kernel
def euclidean_distance(X_train, X_test):
    train_size, feature_size = X_train.shape
    distances = cp.empty(train_size, dtype=cp.float32)
    threads_per_block = 512
    blocks_per_grid = (train_size + threads_per_block - 1) // threads_per_block
    # Launch the kernel
    # the ncu profiling shows memory as bottleneck! 90% mem 9% compute
    with nvtx.annotate("CUDAkernel"):
        euclidean_distance_kernel((blocks_per_grid,), (threads_per_block,), 
                                (X_train, X_test, distances, 
                                cp.int32(train_size), cp.int32(feature_size)))
    
    # Synchronize to ensure the kernel execution is complete
    # cp.cuda.Stream.null.synchronize()
    
    # print(distances)
    return distances

# Predict labels using the CUDA kernel
def predict(X_train, X_test, y_train, n):
    train_size, feature_size = X_train.shape
    test_size = X_test.shape[0]
    max_label = cp.max(y_train).astype(dtype = cp.int32).item()

    y_pred = cp.empty(test_size, dtype=cp.int32)
    indices = cp.arange(train_size, dtype=cp.int32)

    for i in range(test_size):
        distances = euclidean_distance(X_train, X_test[i])
        # threads_per_block = 512
        # blocks_per_grid = (test_size + threads_per_block - 1) // threads_per_block
        
        # predict_kernel((blocks_per_grid,), (threads_per_block,), (y_train, distances, indices, y_pred, n, test_size, train_size, max_label))
        neighbors_idx = cp.argsort(distances)[:n]
        neighbors_labels = y_train[neighbors_idx]
        neighbors_distances = distances[neighbors_idx]
        # predicting using weighted voting according to the closest neighbours
        weights = 1 / (neighbors_distances + 1e-5)  
        weighted_votes = cp.zeros(cp.max(y_train).astype(cp.int64).item() + 1)
        for label, weight in zip(neighbors_labels, weights):
            weighted_votes[label] += weight
        prediction = cp.argmax(weighted_votes)
        
        y_pred[i] = prediction

    return y_pred


# Helper function to run prediction on a specific GPU
def predict_on_gpu(X_train, X_test, y_train, n, device_id, stream):
    with cp.cuda.Device(device_id):
        with stream:
            train_size, feature_size = X_train.shape
            test_size = X_test.shape[0]
            max_label = cp.max(y_train).astype(dtype = cp.int32).item()

            y_pred = cp.empty(test_size, dtype=cp.int32)
            indices = cp.arange(train_size, dtype=cp.int32)

            for i in range(test_size):
                distances = euclidean_distance(X_train, X_test[i])
                # threads_per_block = 512
                # blocks_per_grid = (test_size + threads_per_block - 1) // threads_per_block
                
                # predict_kernel((blocks_per_grid,), (threads_per_block,), (y_train, distances, indices, y_pred, n, test_size, train_size, max_label))
                neighbors_idx = cp.argsort(distances)[:n]
                neighbors_labels = y_train[neighbors_idx]
                neighbors_distances = distances[neighbors_idx]
                # predicting using weighted voting according to the closest neighbours
                weights = 1 / (neighbors_distances + 1e-5)  
                weighted_votes = cp.zeros(cp.max(y_train).astype(cp.int64).item() + 1)
                for label, weight in zip(neighbors_labels, weights):
                    weighted_votes[label] += weight
                prediction = cp.argmax(weighted_votes)
                
                y_pred[i] = prediction

            print(train_size)
            return y_pred

# Splitting data and running prediction on two GPUs
def predict_with_two_gpus(X_train, X_test, y_train, n):
    mid_index = len(X_test) // 2
    X_test_1 = X_test[:mid_index]
    X_test_2 = X_test[mid_index:]

    with cp.cuda.Device(7):
        stream2 = cp.cuda.Stream(non_blocking=True)
        with stream2:
            X_train_gpu_7 = cp.asarray(X_train)
            y_train_gpu_7 = cp.asarray(y_train)
            X_test_2_gpu = cp.asarray(X_test_2)

    with cp.cuda.Device(6):
        stream1 = cp.cuda.Stream(non_blocking=True)
        with stream1:
            X_train_gpu_6 = cp.asarray(X_train)
            y_train_gpu_6 = cp.asarray(y_train)
            X_test_1_gpu = cp.asarray(X_test_1)

    with cp.cuda.Device(6):
        y_pred_1 = predict_on_gpu(X_train_gpu_6, X_test_1_gpu, y_train_gpu_6, n, device_id=6, stream=stream1)

    with cp.cuda.Device(7):
        y_pred_2 = predict_on_gpu(X_train_gpu_7, X_test_2_gpu, y_train_gpu_7, n, device_id=7, stream=stream2)
        
    stream1.synchronize()
    stream2.synchronize()
    # y_pred_2 = cp.asarray(y_pred_2)
    # return cp.concatenate((y_pred_1, y_pred_2))
    y_pred_1_host = y_pred_1.get(stream=stream1)
    y_pred_2_host = y_pred_2.get(stream=stream2)

    y_pred_host = np.concatenate((y_pred_1_host, y_pred_2_host))
    
    return cp.asarray(y_pred_host)

# Add instances where y_pred and y_test are the same and divide by the sample size
def accuracy(y_pred, y_test):
    correct_predictions = cp.sum(y_pred == y_test)
    accuracy_score = correct_predictions / len(y_test)
    return accuracy_score

# Example usage
if __name__ == "__main__":
    cp.cuda.Device(6).use()
    
    with nvtx.annotate("Fetch data"):
        # Traditional mnist dataset downloaded and cached to local directory, (70000, 784), (70000,)
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        
    # mnist dataset fetches target values as object ==  string
    X = cp.asarray(X, dtype = cp.float32) 
    y = cp.asarray(y, dtype = cp.int64) 
    

    # Split the data utilizing sklearn train_test_split
    with nvtx.annotate("Split data"):
        train_indices, test_indices = get_train_test_indices(X.shape[0], test_size=0.25, random_state=42)
        X_train, X_test, y_train, y_test = get_split_arrays(X, y, train_indices, test_indices)
    # Split using self-made algorithm 
    # X_train, X_test, y_train, y_test = train_test_split_gpu(X, y, test_size=0.25, random_state=42)

    start_time = time.time()
    # compute intensive section
    with nvtx.annotate("Predict"):
        y_pred = predict(X_train, X_test, y_train, 3)
        #y_pred = predict_with_two_gpus(X_train, X_test, y_train, 3)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")
    with nvtx.annotate("Accuracy"):
        print(accuracy(y_pred, y_test))
