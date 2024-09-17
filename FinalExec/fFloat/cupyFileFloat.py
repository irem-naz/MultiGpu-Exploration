from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score


import numpy as np
import cupy as cp
import time
import nvtx
import pynvml

class GPUclass:
    def __init__(self):
        self.d_distances = None
        self.d_Xtrain = None
        self.d_Xtest = None
        self.d_yTrain = None
        self.stream = None
        self.memory = None

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

def report_cupy_memory(mempool):
    print(f"Used bytes: {mempool.used_bytes() / (1024 ** 2)} MB")
    print(f"Total bytes: {mempool.total_bytes() / (1024 ** 2)} MB")

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
            myGPUs[i].stream = cp.cuda.Stream(non_blocking=False)
            with myGPUs[i].stream:
                # myGPUs[i].memory = cp.cuda.MemoryAsyncPool()
                # cp.cuda.set_allocator(myGPUs[i].memory.malloc)
                start_idx = i * index
                end_idx = (i + 1) * index if i != GPU_N - 1 else len(X_test)
                myGPUs[i].d_Xtest = cp.asarray(X_test[start_idx:end_idx], dtype=cp.float32, blocking=False)
                myGPUs[i].d_Xtrain = cp.asarray(X_train, dtype=cp.float32, blocking=False)
                myGPUs[i].d_yTrain = cp.asarray(y_train, dtype=cp.int32, blocking=False)
                myGPUs[i].d_distances = cp.empty((index, X_train.shape[0]), dtype=cp.float32)
                # cp.cuda.Device().synchronize()
                # report_cupy_memory(myGPUs[i].memory)

def compute_distances(d_Xtrain, d_Xtest, d_distances):
    d_Xtrain_sq = cp.sum(d_Xtrain ** 2, axis=1)
    d_Xtest_sq = cp.sum(d_Xtest ** 2, axis=1)
    d_distances[:] = cp.sqrt(cp.expand_dims(d_Xtest_sq, axis=1) + d_Xtrain_sq - 2 * cp.dot(d_Xtest, d_Xtrain.T))
    
def split_and_compute_with_gpus(myGPUs, k):
    for i in range(GPU_N):
        with cp.cuda.Device(i):
            with myGPUs[i].stream:
                compute_distances(myGPUs[i].d_Xtrain, myGPUs[i].d_Xtest, myGPUs[i].d_distances)
                # cp.cuda.Device().synchronize()
                # report_cupy_memory(myGPUs[i].memory)
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
                
                # cp.cuda.Device().synchronize()
                # report_cupy_memory(myGPUs[i].memory)

    y_pred = np.concatenate(y_pred_list)
    return y_pred


GPU_N = 1
fraction = 1 

if __name__ == "__main__":
    print("Connecting to GPUs")
    # Accept GPU_N and fraction as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python pureCupy.py <GPU_N> <fraction>")
        sys.exit(1)

    GPU_N = int(sys.argv[1])
    fraction = float(sys.argv[2])
    
    with nvtx.annotate("Fetch data"):
        # X, y = make_classification(n_samples=100000, n_features=100, n_informative=50, n_redundant=10, n_classes=2, random_state=42)
        # print(X.nbytes / 1024 / 1024)
        # print(y.nbytes / 1024 / 1024)
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        # X = np.tile(X, (2, 1))  # Repeat X along the row axis
        # y = np.tile(y, 2) 
        # this would have been the top limit 1.35 7966 MB
        # 7494
        fraction = 0.5

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
        X_resized = X_resized.astype(np.float32)
        y_resized = y_resized.astype(np.int32)

        # Update the original variables
        X = X_resized
        y = y_resized


    start_time = time.time()
    with nvtx.annotate("Get indices"):
        train_indices, test_indices = get_train_test_indices(X.shape[0], test_size=0.25, random_state=42)

    k = 3
    y_test = y[test_indices].astype(np.int64)
    
    print(f"Running with {GPU_N} GPU(s)")
    

    myGPUs = [GPUclass() for _ in range(GPU_N)]

    report_global_mem()
    
    with nvtx.annotate("Initialize Data per GPU"):
        send_data_to_gpus(myGPUs, X, y, train_indices, test_indices)
        report_global_mem()
    
   
    with nvtx.annotate("Calculate Distance"):
        split_and_compute_with_gpus(myGPUs, k)
        # report_global_mem()
    
    with nvtx.annotate("Predict"):
        y_pred = weighted_voting(myGPUs, k)
        # report_global_mem()
        
    end_time = time.time()
    elapsed_time = end_time - start_time

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')


    print(f"Elapsed time with {GPU_N} GPU(s): {elapsed_time} voting")