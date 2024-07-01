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

# Computing distance between values, using euclidean distance
def euclidean_distance(X1, X2):
    return cp.sqrt(cp.sum((X1 - X2) ** 2, axis=1))

# n = nearest neighbour
def predict(X_train, X_test, y_train, n):
    y_pred = cp.empty(len(X_test), dtype=cp.int64)
    for i, x in enumerate(X_test):
        # (56000, 768) x train
        # (14000, 768) x_test
        distances = euclidean_distance(X_train, x)
        neighbors_idx = cp.argsort(distances)[:n]
        neighbors_labels = y_train[neighbors_idx]
        neighbors_distances = distances[neighbors_idx]
        
        # predicting using weighted voting according to the closest neighbours
        weights = 1 / (neighbors_distances + 1e-5)  
        weighted_votes = cp.zeros(cp.max(y_train).item() + 1)
        for label, weight in zip(neighbors_labels, weights):
            weighted_votes[label] += weight
        prediction = cp.argmax(weighted_votes)
        
        y_pred[i] = prediction
    return y_pred


# Helper function to run prediction on a specific GPU
def predict_on_gpu(X_train, X_test, y_train, n, device_id, stream):
    with cp.cuda.Device(device_id):
        with stream:
            y_pred = cp.empty(len(X_test), dtype=cp.float32, stream=stream)
            for i, x in enumerate(X_test):
                distances = euclidean_distance(X_train, x)
                neighbors_idx = cp.argsort(distances)[:n]
                neighbors_labels = y_train[neighbors_idx]
                neighbors_distances = distances[neighbors_idx]
                
                weights = 1 / (neighbors_distances + 1e-5)
                weighted_votes = cp.zeros(cp.max(y_train).item() + 1, stream=stream)
                for label, weight in zip(neighbors_labels, weights):
                    weighted_votes[label] += weight
                prediction = cp.argmax(weighted_votes)
                
                y_pred[i] = prediction
            return y_pred

# Splitting data and running prediction on two GPUs
def predict_with_two_gpus(X_train, X_test, y_train, n):
    mid_index = len(X_test) // 2
    X_test_1 = X_test[:mid_index]
    X_test_2 = X_test[mid_index:]

    with cp.cuda.Device(7):
        stream2 = cp.cuda.Stream(non_blocking=True)
        X_train_gpu_7 = cp.asarray(X_train, stream=stream2)
        y_train_gpu_7 = cp.asarray(y_train, stream=stream2)
        X_test_2_gpu = cp.asarray(X_test_2, stream=stream2)

    with cp.cuda.Device(6):
        stream1 = cp.cuda.Stream(non_blocking=True)
        X_train_gpu_6 = cp.asarray(X_train, stream=stream1)
        y_train_gpu_6 = cp.asarray(y_train, stream=stream1)
        X_test_1_gpu = cp.asarray(X_test_1, stream=stream1)

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
    # y_test = y_test.get()
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
    X = cp.asarray(X) 
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
        # 1 gpu direct execution
        y_pred = predict(X_train, X_test, y_train, 3)
        # 2 gpu execution
        # y_pred = predict_with_two_gpus(X_train, X_test, y_train, 3)
        

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")
    with nvtx.annotate("Accuracy"):
        # y_pred = cp.asarray(y_pred)
        print(f"Accuracy: {accuracy(y_pred, y_test)}")
