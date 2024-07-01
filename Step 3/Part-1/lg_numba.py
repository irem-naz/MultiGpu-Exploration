import numpy as np
import numba 
from numba import cuda
import nvtx
import time
import math


ordinal = 6
device = 'cuda:{ordinal}'.format(ordinal = ordinal)
cuda.select_device(ordinal)

# CUDA kernel for computing cost
@cuda.jit
def compute_cost_kernel(h, y, cost, m):
    shared_temp = cuda.shared.array(256, dtype=numba.float32)
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    loss = 0.0

    if idx < m:
        h_val = h[idx].item()
        y_val = y[idx].item()
        log_h = math.log(h_val)
        log_one_minus_h = math.log(1.0 - h_val)
        loss = -y_val * log_h - (1.0 - y_val) * log_one_minus_h
    
    shared_temp[cuda.threadIdx.x] = loss
    cuda.syncthreads()

    # Reduction to sum up the loss
    i = cuda.blockDim.x // 2
    while i != 0:
        if cuda.threadIdx.x < i:
            shared_temp[cuda.threadIdx.x] += shared_temp[cuda.threadIdx.x + i]
        cuda.syncthreads()
        i //= 2

    if cuda.threadIdx.x == 0:
        cuda.atomic.add(cost, 0, shared_temp[0] / m)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(h, y, m):
    cost = np.zeros(1, dtype=np.float32)
    d_h = cuda.to_device(h)
    d_y = cuda.to_device(y)
    d_cost = cuda.to_device(cost)

    block_dim = 64
    grid_dim = (m + block_dim - 1) // block_dim

    with nvtx.annotate("CUDAkernel"):
        compute_cost_kernel[grid_dim, block_dim](d_h, d_y, d_cost, m)
    
    cost = d_cost.copy_to_host()

    return cost[0]

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    loss_per_step = np.zeros(num_iterations, dtype=np.float32)

    for i in range(num_iterations):
        with nvtx.annotate(f"Iteration {i+1}"):
            with nvtx.annotate("Compute hypothesis"):
                # Compute hypothesis
                h = sigmoid(np.dot(X, theta))
            
            with nvtx.annotate("Compute gradient"):
                # Compute gradient
                gradient = (1/m) * np.dot(X.T, h - y)
                
            with nvtx.annotate("Update weights"):
                # Update weights
                theta -= learning_rate * gradient
            
            with nvtx.annotate("Record Loss"):
                # Record loss
                h = sigmoid(np.dot(X, theta))
                loss_per_step[i] = compute_cost(h, y, m)

    return theta

def main():
    start_time = time.time()
    with nvtx.annotate("Data Generation"):
        n_samples = 10000
        w_true = [-1, 1.2, 1]
        sigma = 0.1
        n_features = 2
        # Generate training data
        X = np.random.uniform(0, 1, size=(n_samples, n_features)).astype(np.float32)
        y = (w_true[0] + w_true[1] * X[:, 0] + w_true[2] * X[:, 1] >= 0).astype(int).reshape(-1, 1).astype(np.float32)
        # Add some noise
        X += sigma * np.random.randn(X.shape[0] * X.shape[1]).reshape(X.shape[0], X.shape[1])
        # Adding intercept column
        X_aug = np.hstack((np.ones((n_samples, 1), dtype=np.float32), X))

    with nvtx.annotate("Initialization"):
        theta = np.zeros((n_features + 1, 1), dtype=np.float32)
        learning_rate = 0.01
        num_iterations = 100

    with nvtx.annotate("Gradient Descent"):
        theta = gradient_descent(X_aug, y, theta, learning_rate, num_iterations)  # Pass X_aug

    print("Theta after gradient descent:", theta)
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()



