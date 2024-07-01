# -*- coding: utf-8 -*-
import cupy as cp
import math
import time

def run_program():
    

    cp.cuda.Device(6).use()

    # Create random input and output data
    x = cp.linspace(-math.pi, math.pi, 2000)
    y = cp.sin(x)

    # Randomly initialize weights
    a = cp.random.randn()
    b = cp.random.randn()
    c = cp.random.randn()
    d = cp.random.randn()

    learning_rate = 1e-6
    start_time = time.time()
    for t in range(2000):
        # Forward pass: compute predicted y
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss
        loss = cp.square(y_pred - y).sum()
        # if t % 100 == 99:
        #     print(t, loss)

        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # Update weights
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time} seconds")
    # print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
    return elapsed_time

# Number of runs
num_runs = 20
total_time = 0.0

for i in range(num_runs):
    total_time += run_program()

average_time = total_time / num_runs
print(f"Average elapsed time over {num_runs} runs: {average_time} seconds")
