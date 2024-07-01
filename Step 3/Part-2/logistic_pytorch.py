import numpy as np
import torch
import time
import nvtx


# Set GPU device
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.matmul(theta))
    cost = (1/m) * torch.sum(-y * torch.log(h) - (1 - y) * torch.log(1 - h))
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    for i in range(num_iterations):

        with nvtx.annotate(f"Iteration {i+1}"):
            with nvtx.annotate("Compute hypothesis"):
                h = sigmoid(X.matmul(theta))
            with nvtx.annotate("Compute gradient"):
                gradient = (1/m) * X.T.matmul(h - y)

            with nvtx.annotate("Update weights"):
                theta -= learning_rate * gradient

            if (i+1) % 100 == 0:
                cost = compute_cost(X, y, theta)
                print(f"Cost after iteration {i+1}: {cost.item()}")

    return theta

def main():
    start_time = time.time()

    with nvtx.annotate("Data Generation"):
        # Data Generation
        np.random.seed(0)
        m = 1000
        n = 2
        X = np.random.rand(m, n).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 1).astype(np.float32)
        y = y.reshape(m, 1)
        X = np.c_[np.ones((m, 1)), X]  # Add intercept term

    with nvtx.annotate("Initialization"):
        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        theta = torch.zeros((n + 1, 1), device=device, dtype=torch.float32)

        learning_rate = 0.01
        num_iterations = 1000

    # Gradient Descent
    # theta = gradient_descent(X, y, theta, learning_rate, num_iterations)
    # Gradient Descent with Profiling
    with nvtx.annotate("Gradient Descent"):
        theta = gradient_descent(X, y, theta, learning_rate, num_iterations)

    print("Theta after gradient descent:", theta)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == "__main__":
    main()
