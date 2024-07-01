import cupy as cp
import numpy as np
import nvtx
import time

# Use GPU device 6
cp.cuda.Device(6).use()

def sigmoid(z):
    return 1 / (1 + cp.exp(-z))

def compute_cost(h, y, theta):
    m = len(y)
    # h = sigmoid(cp.dot(X, theta))
    cost = (1/m) * cp.sum(-y * cp.log(h) - (1 - y) * cp.log(1 - h))
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    loss_per_step = cp.zeros(num_iterations)
    for i in range(num_iterations):
        with nvtx.annotate(f"Iteration {i+1}"):
            with nvtx.annotate("Compute hypothesis"):
                h = sigmoid(X.dot(theta))

            with nvtx.annotate("Compute gradient"):
                gradient = (1/m) * X.T.dot(h - y)

            with nvtx.annotate("Update weights"):
                theta -= learning_rate * gradient

            with nvtx.annotate("Record Loss"):
                # loss_per_step[i] = compute_cost(X, y, theta)
                h = sigmoid(cp.dot(X, theta))
                loss_per_step[i] = compute_cost(h, y, m) 
                # print(f"Cost after iteration {i+1}: {loss_per_step[i]}")

    return theta

def main():

    start_time = time.time()

    
    with nvtx.annotate("Data Generation"):
        cp.random.seed(0)
        m = 1000
        n = 2
        X = cp.random.rand(m, n)
        y = (X[:, 0] + X[:, 1] > 1).astype(cp.float32)
        y = y.reshape(m, 1)
        X = cp.c_[cp.ones((m, 1)), X]  # Add intercept term

    
    with nvtx.annotate("Initialization"):
        # X = cp.asarray(X)
        # y = cp.asarray(y)
        theta = cp.zeros((n + 1, 1))
        learning_rate = 0.01
        num_iterations = 1000


    # with nvtx.annotate("Data Generation"):
    #     n_samples = 10000
    #     w_true = [-1, 1.2, 1]
    #     sigma = 0.1
    #     n_features = 2
    #     # Generate training data
    #     X = cp.random.uniform(0, 1, size=(n_samples, n_features))
    #     y = cp.array(w_true[0] + w_true[1] * X[:, 0] + w_true[2] * X[:, 1] >= 0).astype(int).reshape(-1, 1)
    #     # Add some noise
    #     X = X + sigma * cp.random.randn(X.shape[0] * X.shape[1]).reshape(X.shape[0], X.shape[1])
    #     # Adding intercept column
    #     X_aug = cp.hstack((cp.ones((n_samples, 1)), X))

    # with nvtx.annotate("Initialization"):
    #     theta = cp.zeros((n_features + 1, 1))
    #     learning_rate = 0.01
    #     num_iterations = 1000
    
    with nvtx.annotate("Gradient Descent"):
        theta = gradient_descent(X, y, theta, learning_rate, num_iterations)

    print("Theta after gradient descent:", theta)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()
