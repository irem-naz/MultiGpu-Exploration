import cupy as cp
import numpy as np
import nvtx
import time
import matplotlib.pyplot as plt

# Set Matplotlib backend to Agg
import matplotlib
matplotlib.use('Agg')

# Set GPU device 3 for CuPy
cp.cuda.Device(3).use()

def sigmoid(z, lib):
    return 1 / (1 + lib.exp(-z))

def compute_cost(X, y, theta, lib):
    m = len(y)
    h = sigmoid(X.dot(theta), lib)
    cost = (1/m) * lib.sum(-y * lib.log(h) - (1 - y) * lib.log(1 - h))
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations, lib, lib_name):
    m = len(y)
    for i in range(num_iterations):
        with nvtx.annotate(f"Iteration {i+1} ({lib_name})"):
            with nvtx.annotate("Compute hypothesis"):
                h = sigmoid(X.dot(theta), lib)

            with nvtx.annotate("Compute gradient"):
                gradient = (1/m) * X.T.dot(h - y)

            with nvtx.annotate("Update weights"):
                theta -= learning_rate * gradient

            if (i+1) % 100 == 0:
                cost = compute_cost(X, y, theta, lib)
                print(f"Cost after iteration {i+1} ({lib_name}): {cost}")

    return theta

def run_experiment(m, n, num_iterations=1000, learning_rate=0.01, lib=cp, lib_name="CuPy"):
    with nvtx.annotate(f"Experiment ({lib_name})"):
        start_time = time.time()

        # Generate some synthetic data
        with nvtx.annotate("Data Generation"):
            lib.random.seed(0)
            X = lib.random.rand(m, n)
            y = (X[:, 0] + X[:, 1] > 1).astype(lib.float32)
            y = y.reshape(m, 1)
            X = lib.c_[lib.ones((m, 1)), X]  # Add intercept term

        # Initialize parameters
        with nvtx.annotate("Initialization"):
            theta = lib.zeros((n + 1, 1))

        # Perform gradient descent
        with nvtx.annotate("Gradient Descent"):
            theta = gradient_descent(X, y, theta, learning_rate, num_iterations, lib, lib_name)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for m={m}, n={n} ({lib_name}): {elapsed_time} seconds")
        
        return elapsed_time


def main():
    num_iterations = 1000
    learning_rate = 0.01
    n = 2  # Number of features

    data_sizes = [100, 1000, 5000, 10000, 20000, 40000, 80000]  # Different sizes of synthetic data
    times_cupy = []
    times_numpy = []


    print("Running CuPy experiments:")
    for m in data_sizes:
        elapsed_time = run_experiment(m, n, num_iterations, learning_rate, cp, "CuPy")
        times_cupy.append(elapsed_time)

    print("Running NumPy experiments:")
    for m in data_sizes:
        elapsed_time = run_experiment(m, n, num_iterations, learning_rate, np, "NumPy")
        times_numpy.append(elapsed_time)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, times_cupy, label="CuPy", marker='o')
    plt.plot(data_sizes, times_numpy, label="NumPy", marker='s')
    # plt.plot(data_sizes, times_numpy, label="NumPy", marker='s')

    plt.xlabel("Data Size")
    plt.ylabel("Elapsed Time (seconds)")
    plt.title("Elapsed Time vs Data Size for CuPy and NumPy")
    plt.legend()
    plt.grid(True)
    plt.savefig("elapsed_time_vs_data_size.png")
    print("Plot saved as 'elapsed_time_vs_data_size.png'.")

if __name__ == "__main__":
    main()



# import cupy as cp
# import nvtx
# import time

# # Set GPU device 3
# cp.cuda.Device(3).use()

# def sigmoid(z):
#     return 1 / (1 + cp.exp(-z))

# def compute_cost(X, y, theta):
#     m = len(y)
#     h = sigmoid(X.dot(theta))
#     cost = (1/m) * cp.sum(-y * cp.log(h) - (1 - y) * cp.log(1 - h))
#     return cost

# def gradient_descent(X, y, theta, learning_rate, num_iterations):
#     m = len(y)
#     for i in range(num_iterations):
#         with nvtx.annotate(f"Iteration {i+1}"):
#             with nvtx.annotate("Compute hypothesis"):
#                 h = sigmoid(X.dot(theta))

#             with nvtx.annotate("Compute gradient"):
#                 gradient = (1/m) * X.T.dot(h - y)

#             with nvtx.annotate("Update weights"):
#                 theta -= learning_rate * gradient

#             if (i+1) % 100 == 0:
#                 cost = compute_cost(X, y, theta)
#                 print(f"Cost after iteration {i+1}: {cost}")

#     return theta

# def run_experiment(m, n, num_iterations=1000, learning_rate=0.01):
#     with nvtx.annotate("Experiment"):
#         start_time = time.time()

#         # Generate some synthetic data
#         with nvtx.annotate("Data Generation"):
#             cp.random.seed(0)
#             X = cp.random.rand(m, n)
#             y = (X[:, 0] + X[:, 1] > 1).astype(cp.float32)
#             y = y.reshape(m, 1)
#             X = cp.c_[cp.ones((m, 1)), X]  # Add intercept term

#         # Initialize parameters
#         with nvtx.annotate("Initialization"):
#             theta = cp.zeros((n + 1, 1))

#         # Perform gradient descent
#         with nvtx.annotate("Gradient Descent"):
#             theta = gradient_descent(X, y, theta, learning_rate, num_iterations)

#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         print(f"Elapsed time for m={m}, n={n}: {elapsed_time} seconds")
        
#         return elapsed_time

# def main():
#     num_iterations = 1000
#     learning_rate = 0.01
#     n = 2  # Number of features

#     data_sizes = [100, 1000, 5000, 10000, 20000]  # Different sizes of synthetic data
#     times = []

#     for m in data_sizes:
#         elapsed_time = run_experiment(m, n, num_iterations, learning_rate)
#         times.append(elapsed_time)

#     # Print or plot the results
#     print("Data sizes:", data_sizes)
#     print("Elapsed times:", times)

# if __name__ == "__main__":
#     main()
