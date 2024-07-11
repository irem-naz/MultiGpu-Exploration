from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

import numpy as np
import cupy as cp
import time
import nvtx

class GPUclass:
    def __init__(self):
        self.d_distances = None
        self.d_Xtrain = None
        self.d_Xtest = None
        self.d_yTrain = None
        self.stream = None

# Using train_test_split algorithm to only get the indices using CPU
def get_train_test_indices(n_samples, test_size=0.25, random_state=None):
    indices = np.arange(n_samples)
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    return train_indices, test_indices


def send_data_to_gpus(myGPUs, X, y, train_indices, test_indices):
    X_test = X[test_indices]
    X_train = X[train_indices]
    y_train = y[train_indices]

    index = len(X_test) // GPU_N
    for i in range(GPU_N):
        with cp.cuda.Device(i):
            myGPUs[i].stream = cp.cuda.Stream(non_blocking = True)
            with myGPUs[i].stream:
                start_idx = i * index
                end_idx = (i + 1) * index if i != GPU_N - 1 else len(X_test)
                myGPUs[i].d_Xtest = cp.asarray(X_test[start_idx:end_idx], dtype=cp.float32, blocking=False)
                myGPUs[i].d_Xtrain = cp.asarray(X_train, dtype=cp.float32, blocking=False)
                myGPUs[i].d_yTrain = cp.asarray(y_train, dtype=cp.int32, blocking=False)
                myGPUs[i].d_distances = cp.empty((index, X_train.shape[0]), dtype=cp.float32)


euclidean_distance_kernel = cp.RawKernel(r'''
extern "C" __global__
void euclidean_distance_kernel(float* X_train, float* X_test, float* distances, int train_size, int test_size, int feature_size) {
    int train_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Index for X_train
    int test_idx = blockIdx.y;  // Index for X_test

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
''', 'euclidean_distance_kernel')

def split_and_compute_with_gpus(myGPUs, k):
    
    train_size = myGPUs[0].d_Xtrain.shape[0]
    test_size = myGPUs[0].d_Xtest.shape[0]
    feature_size = myGPUs[0].d_Xtest.shape[1]

    # Kernel parameters
    threads_per_block = 256
    blocks_per_grid_x = (train_size + threads_per_block - 1) // threads_per_block
    blocks_per_grid_y = test_size
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)


    for i in range(GPU_N):
        with cp.cuda.Device(i):
            with myGPUs[i].stream:
                euclidean_distance_kernel(
                    blocks_per_grid, (threads_per_block,),
                    (myGPUs[i].d_Xtrain, myGPUs[i].d_Xtest, myGPUs[i].d_distances, cp.int32(train_size), cp.int32(test_size), cp.int32(feature_size))
                    , stream=myGPUs[i].stream)


def weighted_voting(myGPUs, k):
    y_pred_list = []
    for i in range(GPU_N):
        with cp.cuda.Device(i):
            with myGPUs[i].stream:
                distances = myGPUs[i].d_distances
                y_train = myGPUs[i].d_yTrain
                
                # Get the indices of the k smallest distances
                neighbors_idx = cp.argsort(distances, axis=1)[:, :k]
                # Expand dimensions of y_train to match neighbors_idx
                y_train_expanded = cp.expand_dims(y_train, axis=0)

                # Gather the corresponding labels and distances
                neighbors_labels = cp.take_along_axis(y_train_expanded, neighbors_idx, axis=1).squeeze()
                neighbors_distances = cp.take_along_axis(distances, neighbors_idx, axis=1)
                
                # Predicting using weighted voting according to the closest neighbors
                weights = 1 / (neighbors_distances + 1e-5)
                max_label = cp.max(y_train).astype(cp.int32).item() + 1
                weighted_votes = cp.zeros((distances.shape[0], max_label))

                # Vectorized weighted voting
                for j in range(k):
                    labels = neighbors_labels[:, j]
                    weight = weights[:, j]
                    for row in range(distances.shape[0]):
                        weighted_votes[row, labels[row]] += weight[row]
                predictions = cp.argmax(weighted_votes, axis=1)
                
                # Append predictions to the list
                y_pred_list.append(predictions.get())

    y_pred = np.concatenate(y_pred_list)
    return y_pred



GPU_N = 8

# Example usage
if __name__ == "__main__":
    print("Connecting to GPUs")
    cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)
    
    
    with nvtx.annotate("Fetch data"):
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    with nvtx.annotate("Get indices"):
        # indices as np.array
        train_indices, test_indices = get_train_test_indices(X.shape[0], test_size=0.25, random_state=42)

    k = 3
    
    y_test = y[test_indices].astype(np.int32)
    
    for GPU_N in range(1, 9):
        print(f"Running with {GPU_N} GPU(s)")
        
        # Initialize structs per GPU
        myGPUs = [GPUclass() for _ in range(GPU_N)]

        num_runs = 10
        total_time = 0
        
        for _ in range(num_runs):
    
            start_time = time.time()

            with nvtx.annotate("Initialize Data per GPU"):
                send_data_to_gpus(myGPUs, X, y, train_indices, test_indices)
            

            with nvtx.annotate("Calculate Distance"):
                split_and_compute_with_gpus(myGPUs, k)

            
            with nvtx.annotate("Predict"):
                y_pred = weighted_voting(myGPUs, k)
                
            end_time = time.time()
            elapsed_time = end_time - start_time

            total_time += elapsed_time
        
        average_time = total_time / num_runs

        print(f"Elapsed time with {GPU_N} GPU(s) over {num_runs} runs: {average_time} seconds")
