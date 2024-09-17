from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


import numpy as np
import cupy as cp
import time
import nvtx
import pynvml
import sys



class GPUclass:
    def __init__(self):
        self.d_distances = None
        self.d_Xtrain = None
        self.d_Xtest = None
        self.d_yTrain = None
        self.stream = None
        self.memory = None
        self.predictions = None

def report_global_mem():
    pynvml.nvmlInit()
    for i in range(GPU_N):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU {i}:")
        print(f"  Total memory: {mem_info.total / (1024 ** 2)} MB")
        print(f"  Free memory: {mem_info.free / (1024 ** 2)} MB")
        print(f"  Used memory: {mem_info.used / (1024 ** 2)} MB")
    pynvml.nvmlShutdown()

# def report_cupy_memory(mempool):
#     cp.cuda.Device().synchronize()  # Ensure all operations are complete
#     print(f"Used bytes: {mempool.used_bytes() / (1024 ** 2)} MB")
#     print(f"Total bytes: {mempool.total_bytes() / (1024 ** 2)} MB")

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
            myGPUs[i].stream = cp.cuda.Stream(non_blocking=True)
            with myGPUs[i].stream:
                # myGPUs[i].memory = cp.cuda.MemoryAsyncPool()
                # cp.cuda.set_allocator(myGPUs[i].memory.malloc)
                start_idx = i * index
                end_idx = (i + 1) * index if i != GPU_N - 1 else len(X_test)
                myGPUs[i].d_Xtest = cp.asarray(X_test[start_idx:end_idx], dtype=cp.int32, blocking=False)
                myGPUs[i].d_Xtrain = cp.asarray(X_train, dtype=cp.int32, blocking=False)
                myGPUs[i].d_yTrain = cp.asarray(y_train, dtype=cp.int32, blocking=False)
                myGPUs[i].d_distances = cp.empty((index, X_train.shape[0]), dtype=cp.float32)
                myGPUs[i].predictions = cp.empty(index, dtype=cp.int32)
                # cp.cuda.Device().synchronize()
                # report_cupy_memory(myGPUs[i].memory)

euclidean_distance_kernel = cp.RawKernel(r'''
extern "C" __global__
void euclidean_distance_kernel(int* X_train, int* X_test, float* distances, int train_size, int test_size, int feature_size) {
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

    threads_per_block = 256
    blocks_per_grid_x = (train_size + threads_per_block - 1) // threads_per_block
    blocks_per_grid_y = test_size
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    for i in range(GPU_N):
        with cp.cuda.Device(i):
            with myGPUs[i].stream:
                euclidean_distance_kernel(
                    blocks_per_grid, (threads_per_block,),
                    (myGPUs[i].d_Xtrain, myGPUs[i].d_Xtest, myGPUs[i].d_distances, cp.int32(train_size), cp.int32(test_size), cp.int32(feature_size)),
                    stream=myGPUs[i].stream
                )

def weighted_voting_kernel_call(myGPUs, k):
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
                
                # cp.cuda.Device().synchronize()
                # report_cupy_memory(myGPUs[i].memory)

    y_pred = np.concatenate(y_pred_list)
    return y_pred

weighted_voting_kernel = cp.RawKernel(r'''
extern "C" __global__
void weighted_voting_kernel(float* distances, int* y_train, int* predictions, int test_size, int train_size, int k, int max_label, float* neighbors_distances, int* neighbors_labels) {
    int test_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Index for test samples
    if (test_idx < test_size) {
        // Pointers to the neighbors' distances and labels for this test sample
        float* local_neighbors_distances = &neighbors_distances[test_idx * k];
        int* local_neighbors_labels = &neighbors_labels[test_idx * k];

        // Initialize nearest neighbors
        for (int i = 0; i < k; i++) {
            local_neighbors_distances[i] = 1e10;
            local_neighbors_labels[i] = -1;
        }

        // Find k nearest neighbors
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

       // Perform weighted voting
        float weighted_votes[10];  // Assuming max_label is 10
        for (int i = 0; i < max_label; i++) {
            weighted_votes[i] = 0.0;
        }

        for (int i = 0; i < k; i++) {
            weighted_votes[local_neighbors_labels[i]] += 1.0 / (local_neighbors_distances[i] + 1e-5);
        }

        // Get prediction
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
''', 'weighted_voting_kernel')

def weighted_voting(myGPUs, k):
    y_pred_list = []
    for i in range(GPU_N):
        with cp.cuda.Device(i):
            with myGPUs[i].stream:
                
                test_size = myGPUs[i].d_distances.shape[0]
                train_size = myGPUs[i].d_distances.shape[1]
                max_label = cp.max(myGPUs[i].d_yTrain).astype(cp.int32).item() + 1

                
                
                # Allocate global memory for neighbors' distances and labels
                neighbors_distances = cp.zeros((test_size, k), dtype=cp.float32)
                neighbors_labels = cp.zeros((test_size, k), dtype=cp.int32)

                threads_per_block = 256
                blocks_per_grid = (test_size + threads_per_block - 1) // threads_per_block

                weighted_voting_kernel(
                    (blocks_per_grid,), (threads_per_block,),
                    (myGPUs[i].d_distances, myGPUs[i].d_yTrain, myGPUs[i].predictions, cp.int32(test_size), cp.int32(train_size), cp.int32(k), cp.int32(max_label),
                    neighbors_distances, neighbors_labels),
                    stream=myGPUs[i].stream
                )

            # cp.cuda.Stream().synchronize()  

                # y_pred = predictions.get()

    # y_pred = weighted_voting_kernel_call(myGPUs, k)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {accuracy * 100:.2f}%')
    # print(y_test)
    # return y_pred


def get_predictions(myGPUs):
    y_pred_list = []
    for i in range(GPU_N):
        with cp.cuda.Device(i):
            with myGPUs[i].stream:
                y_pred_list.append(myGPUs[i].predictions.get())

    y_pred = np.concatenate(y_pred_list)
    return y_pred

GPU_N = 1
fraction = 1

if __name__ == "__main__":
    print("Connecting to GPUs")
    # cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)
    
    # Accept GPU_N and fraction as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python pureCupy.py <GPU_N> <fraction>")
        sys.exit(1)

    GPU_N = int(sys.argv[1])
    fraction = float(sys.argv[2])
    
    with nvtx.annotate("Fetch data"):
        # float 64
        # X, y = make_classification(n_samples=100000, n_features=786, n_informative=50, n_redundant=10, n_classes=2, random_state=42)
        # print(X.nbytes/1024/1024)
        # print(y.nbytes/1024/1024)
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        # Define the fraction to increase the size by (e.g., 1.5 means increase by 50%)
        # this would have been the top limit is 1.52 9496 MB
        # 1 2.8 3.6 4

        # Calculate the new number of rows
        original_num_rows = X.shape[0]
        new_num_rows = int(original_num_rows * fraction)

        # Use np.tile to repeat the original array
        num_repeats = int(np.ceil(fraction))
        tiled_X = np.tile(X, (num_repeats, 1))
        tiled_y = np.tile(y, num_repeats)

        # Slice to get the desired number of rows
        X_resized = tiled_X[:new_num_rows]
        y_resized = tiled_y[:new_num_rows]

        # Convert data types
        X_resized = X_resized.astype(np.int32)
        y_resized = y_resized.astype(np.int32)

        # Update the original variables
        X = X_resized
        y = y_resized

    # Print the new shapes
    print("New shape:", X.shape)
    print("New shape of y:", y.shape)
        

    start_time = time.time() 
    with nvtx.annotate("Get indices"):
        # indices as np.array
        train_indices, test_indices = get_train_test_indices(X.shape[0], test_size=0.25, random_state=42)

    k = 3
    
    y_test = y[test_indices]
    
    
    print(f"Running with {GPU_N} GPU(s)")
    
    # Initialize structs per GPU
    myGPUs = [GPUclass() for _ in range(GPU_N)]

  

    report_global_mem()
    

    

    with nvtx.annotate("Initialize Data per GPU"):
        send_data_to_gpus(myGPUs, X, y, train_indices, test_indices)
        report_global_mem()
    
    

    with nvtx.annotate("Calculate Distance"):
        split_and_compute_with_gpus(myGPUs, k)
        # report_global_mem()

    
    with nvtx.annotate("Predict"):
        weighted_voting(myGPUs, k)

    y_pred = get_predictions(myGPUs)
    # print(y_pred)
    # print(y_test)
        
    end_time = time.time()
    elapsed_time = end_time - start_time

    


    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')


    print(f"Elapsed time with {GPU_N} GPU(s): {elapsed_time} seconds")