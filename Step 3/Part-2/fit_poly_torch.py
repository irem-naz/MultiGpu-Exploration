# -*- coding: utf-8 -*-
import torch
import math
import time

def run_program():
    start_time = time.time()

    dtype = torch.float
    device = torch.device("cuda:7")  # Ensure that the GPU is being used

    # Create random input and output data on the GPU
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # Randomly initialize weights on the GPU
    a = torch.randn((), device=device, dtype=dtype)
    b = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)
    d = torch.randn((), device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(2000):
        # Forward pass: compute predicted y
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum()
        # if t % 100 == 99:
        #     print(t, loss.item())

        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # Update weights using gradient descent
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time} seconds")
    # print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
    
    # Verify if the tensors are on the GPU
    # print(f"x device: {x.device}, y device: {y.device}")
    # print(f"a device: {a.device}, b device: {b.device}, c device: {c.device}, d device: {d.device}")

    return elapsed_time

# Number of runs
num_runs = 20
total_time = 0.0

for i in range(num_runs):
    # print(f"Run {i + 1}/{num_runs}")
    total_time += run_program()

average_time = total_time / num_runs
print(f"Average elapsed time over {num_runs} runs: {average_time} seconds")
