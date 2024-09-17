import numpy as np
from sklearn.datasets import fetch_openml

# Fetch the data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Define the fraction to increase the size by (e.g., 1.6 means increase by 60%)
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
# takes circa 2 seconds
X_resized = X_resized.astype(np.float32)
y_resized = y_resized.astype(np.int32)

# Convert data types
# takes circa 13.6 seconds
# X_resized = X_resized.astype(np.int32)
# y_resized = y_resized.astype(np.int32)

print(X_resized.shape)
print(y_resized.shape)

# Save the data to a file, including the shape
with open('mnist_data.bin', 'wb') as f:
    np.array(X_resized.shape, dtype=np.int32).tofile(f)
    X_resized.tofile(f)
    np.array(y_resized.shape, dtype=np.int32).tofile(f)
    y_resized.tofile(f)


# import numpy as np
# from sklearn.datasets import fetch_openml

# # Fetch the data
# X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# # Convert data to appropriate types
# X = X.astype(np.float32)
# y = y.astype(np.int32)

# # Save the data to a file, including the shape
# with open('mnist_data.bin', 'wb') as f:
#     np.array(X.shape, dtype=np.int32).tofile(f)
#     X.tofile(f)
#     np.array(y.shape, dtype=np.int32).tofile(f)
#     y.tofile(f)







# import cupy as cp
# import numpy as np
# import time
# import pynvml
# import nvtx

# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.datasets import make_classification



# class GPUclass:
#     def __init__(self):
#         self.d_distances = None
#         self.d_Xtrain = None
#         self.d_Xtest = None
#         self.d_yTrain = None
#         self.stream = None
#         self.memory = None



# euclidean_distance_kernel = cp.RawKernel(r'''
# extern "C" __global__
# void euclidean_distance_kernel(float* X_train, float* X_test, float* distances, int train_size, int test_size, int feature_size) {
#     int train_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Index for X_train
#     int test_idx = blockIdx.y;  // Index for X_test

#     if (train_idx < train_size && test_idx < test_size) {
#         float dist = 0.0;
#         for (int i = 0; i < feature_size; i++) {
#             float train_val = X_train[train_idx * feature_size + i];
#             float test_val = X_test[test_idx * feature_size + i];
#             float diff = train_val - test_val;
#             dist += diff * diff;
#         }
#         distances[test_idx * train_size + train_idx] = sqrtf(dist);
#     }
# }
# ''', 'euclidean_distance_kernel')




# def split_and_compute_with_gpus(myGPU, k):
#     train_size = myGPU.d_Xtrain.shape[0]
#     test_size = myGPU.d_Xtest.shape[0]
#     feature_size = myGPU.d_Xtest.shape[1]

#     # Kernel parameters
#     threads_per_block = 256
#     blocks_per_grid_x = (train_size + threads_per_block - 1) // threads_per_block
#     blocks_per_grid_y = test_size
#     blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

#     with myGPU.stream:
#         euclidean_distance_kernel(
#             blocks_per_grid, (threads_per_block,),
#             (myGPU.d_Xtrain, myGPU.d_Xtest, myGPU.d_distances, cp.int32(train_size), cp.int32(test_size), cp.int32(feature_size)),
#             stream=myGPU.stream
#         )
#         # myGPU.stream.synchronize()
#             # Launch the host callback function in the stream
#             # myGPUs[i].stream.launch_host_func(host_callback, i)


# def weighted_voting_kernel_call(myGPU, k):
#     distances = myGPU.d_distances
#     y_train = myGPU.d_yTrain

#     # Get the indices of the k smallest distances
#     neighbors_idx = cp.argsort(distances, axis=1)[:, :k]
#     # Expand dimensions of y_train to match neighbors_idx
#     y_train_expanded = cp.expand_dims(y_train, axis=0)

#     # Gather the corresponding labels and distances
#     neighbors_labels = cp.take_along_axis(y_train_expanded, neighbors_idx, axis=1).squeeze()
#     neighbors_distances = cp.take_along_axis(distances, neighbors_idx, axis=1)

#     # Predicting using weighted voting according to the closest neighbors
#     weights = 1 / (neighbors_distances + 1e-5)
#     max_label = cp.max(y_train).astype(cp.int32).item() + 1
#     weighted_votes = cp.zeros((distances.shape[0], max_label))

#     # Vectorized weighted voting
#     for j in range(k):
#         labels = neighbors_labels[:, j]
#         weight = weights[:, j]
#         for row in range(distances.shape[0]):
#             weighted_votes[row, labels[row]] += weight[row]
    
#     print(weighted_votes.shape)
#     predictions = cp.argmax(weighted_votes, axis=1)

#     return predictions.get()

# weighted_voting_kernel = cp.RawKernel(r'''
# extern "C" __global__
# void weighted_voting_kernel(float* distances, int* y_train, int* predictions, int test_size, int train_size, int k, int max_label, float* neighbors_distances, int* neighbors_labels) {
#     int test_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Index for test samples
#     if (test_idx < test_size) {
#         // Pointers to the neighbors' distances and labels for this test sample
#         float* local_neighbors_distances = &neighbors_distances[test_idx * k];
#         int* local_neighbors_labels = &neighbors_labels[test_idx * k];

#         // Initialize nearest neighbors
#         for (int i = 0; i < k; i++) {
#             local_neighbors_distances[i] = 1e10;
#             local_neighbors_labels[i] = -1;
#         }

#         // Find k nearest neighbors
#         for (int i = 0; i < train_size; i++) {
#             float dist = distances[test_idx * train_size + i];
#             for (int j = 0; j < k; j++) {
#                 if (dist < local_neighbors_distances[j]) {
#                     for (int l = k - 1; l > j; l--) {
#                         local_neighbors_distances[l] = local_neighbors_distances[l - 1];
#                         local_neighbors_labels[l] = local_neighbors_labels[l - 1];
#                     }
#                     local_neighbors_distances[j] = dist;
#                     local_neighbors_labels[j] = y_train[i];
#                     break;
#                 }
#             }
#         }

#        // Perform weighted voting
#         float weighted_votes[2];  // Assuming max_label is 10
#         for (int i = 0; i < max_label; i++) {
#             weighted_votes[i] = 0.0;
#         }

#         for (int i = 0; i < k; i++) {
#             weighted_votes[local_neighbors_labels[i]] += 1.0 / (local_neighbors_distances[i] + 1e-5);
#         }

#         // Get prediction
#         float max_vote = 0.0;
#         int pred_label = -1;
#         for (int i = 0; i < max_label; i++) {
#             if (weighted_votes[i] > max_vote) {
#                 max_vote = weighted_votes[i];
#                 pred_label = i;
#             }
#         }
#         predictions[test_idx] = pred_label; 
#     }
# }
# ''', 'weighted_voting_kernel')

# def weighted_voting(myGPU, k):            
#     with myGPU.stream:
#         # y_pred = weighted_voting_kernel_call(myGPU, k)

#         distances = myGPU.d_distances
#         y_train = myGPU.d_yTrain
#         test_size = distances.shape[0]
#         train_size = distances.shape[1]
#         max_label = cp.max(y_train).astype(cp.int32).item() + 1

#         predictions = cp.empty(test_size, dtype=cp.int32)
        
#         # Allocate global memory for neighbors' distances and labels
#         neighbors_distances = cp.zeros((test_size, k), dtype=cp.float32)
#         neighbors_labels = cp.zeros((test_size, k), dtype=cp.int32)

#         threads_per_block = 32
#         blocks_per_grid = (test_size + threads_per_block - 1) // threads_per_block
        
#         # myGPU.stream.synchronize()

#         weighted_voting_kernel(
#             (blocks_per_grid,), (threads_per_block,),
#             (myGPU.d_distances, myGPU.d_yTrain, predictions, cp.int32(test_size), cp.int32(train_size), cp.int32(k), cp.int32(max_label),
#             neighbors_distances, neighbors_labels),
#             stream=myGPU.stream
#         )

#         # myGPU.stream.synchronize()

#         y_pred = predictions.get()

#     y_test = weighted_voting_kernel_call(myGPU, k)

#     accuracy = accuracy_score(y_test, y_pred)
#     print(f'Accuracy: {accuracy * 100:.2f}%')

#     return y_pred



# # Example usage
# if __name__ == "__main__":
#     cp.cuda.Device(1).use()

#     myGPU = GPUclass()
#     myGPU.stream = cp.cuda.Stream(non_blocking=True)

#     print("Connecting to GPUs")
#     with nvtx.annotate("Fetch data"):
#         X, y = make_classification(n_samples=200, n_features=50, n_informative=2, n_classes=2, random_state=42)
        
#     with nvtx.annotate("Get indices"):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         X_train.astype(np.float32)
#         X_test.astype(np.float32)
#         y_train.astype(np.int32)
#         y_test.astype(np.int32)
#     myGPU.memory = cp.cuda.MemoryAsyncPool()
#     cp.cuda.set_allocator(myGPU.memory.malloc)

#     myGPU.d_Xtest = cp.asarray(X_test, dtype=cp.float32, blocking=False)
#     myGPU.d_Xtrain = cp.asarray(X_train, dtype=cp.float32, blocking=False)
#     myGPU.d_yTrain = cp.asarray(y_train, dtype=cp.int32, blocking=False)
#     myGPU.d_distances = cp.empty((X_test.shape[0], X_train.shape[0]), dtype=cp.float32)
#     k = 3


#     start_time = time.time()
#     print("Calculate Distance")
#     split_and_compute_with_gpus(myGPU, k)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     # print(f"Elapsed time: {elapsed_time} for compute")

#     start_time = time.time()
#     print("Predict")
#     y_pred = weighted_voting(myGPU, k)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Elapsed time: {elapsed_time} for voting")

   
    

#     # Print results
