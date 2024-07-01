import cupy as cp
import numpy as np
import nvtx
import time
import timeit
import matplotlib.pyplot as plt
import seaborn as sns

# CUDA kernel as a string
compute_cost_kernel_code = '''
extern "C" __global__
void compute_cost_kernel(const float* h, const float* y, float* cost, int m) {
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
''' 

compute_cost_kernel = cp.RawKernel(compute_cost_kernel_code, 'compute_cost_kernel')


# Use GPU
cp.cuda.Device(6).use()

def sigmoid(z):
    return 1 / (1 + cp.exp(-z))

def compute_cost(h, y, m):
    # cost = (1/m) * cp.sum(-y * cp.log(h) - (1 - y) * cp.log(1 - h))  
    # Using CUDA kernel
    cost = cp.zeros(1, dtype=cp.float32)
    
    # Grid and block dimensions
    block_dim = 64
    grid_dim = (m + block_dim - 1) // block_dim
    
    # Launching!
    with nvtx.annotate("CUDAkernel"):
        compute_cost_kernel((grid_dim,), (block_dim,), (h, y, cost, m))
    
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)  
    loss_per_step = cp.zeros(num_iterations)  

    for i in range(num_iterations):
        with nvtx.annotate(f"Iteration {i+1}"):

            #Compute prediction
            with nvtx.annotate("Compute hypothesis"):
                h = sigmoid(cp.dot(X, theta))

            #Calculate gradient
            with nvtx.annotate("Compute gradient"):
                gradient = (1/m) * cp.dot(X.T, y - h)  

            #Calculate new theta
            with nvtx.annotate("Update weights"):
                theta += learning_rate * gradient  

            #Loss should keep decreasing in a good model
            with nvtx.annotate("Record Loss"):
                h = sigmoid(cp.dot(X, theta))
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
        X = cp.random.uniform(0, 1, size=(n_samples, n_features))
        y = cp.array(w_true[0] + w_true[1] * X[:, 0] + w_true[2] * X[:, 1] >= 0).astype(int).reshape(-1, 1)
        # Add some noise
        X = X + sigma * cp.random.randn(X.shape[0] * X.shape[1]).reshape(X.shape[0], X.shape[1])
        # Adding intercept column
        X_aug = cp.hstack((cp.ones((n_samples, 1)), X))

    with nvtx.annotate("Initialization"):
        theta = cp.zeros((n_features + 1, 1))
        learning_rate = 0.01
        num_iterations = 100

    with nvtx.annotate("Gradient Descent"):
        theta = gradient_descent(X_aug, y, theta, learning_rate, num_iterations)  # Pass X_aug

    print("Theta after gradient descent:", theta)
    # plt.subplot(1,3, 2)
    # sns.lineplot(x=cp.arange(num_iterations), y=loss_per_step[0:num_iterations])
    # plt.yscale("log")
    # plt.xlabel("Iteration")
    # plt.ylabel("Cross Entropy Loss")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("cross_loss.png")
    # print("Plot saved as 'cross_loss.png'.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")



if __name__ == "__main__":
    main()
