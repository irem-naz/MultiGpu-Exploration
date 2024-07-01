# Comparing Frameworks

There are 3 frameworks in Python language that allows low-level kernel configuration management: Cupy - a drop-in replacement for Numpy -, Numba - which is JIT compiler for CUDA option -, and Pytorch which can launch its arrays in GPU due to inherent GPU acceleration and allows kernel management through C to Python translation libraries such as Pybind11.

The same binary logistic regression is implemented for each of the models, the data is generated as part of the program.

## [Part-1: CUDA vs Cupy vs Numba](./Part-1)
**!** Since there were many problems faced in implementing Pybind11, Pytorch is not compared to other frameworks here that use CUDA kernels to manage kernel launch configurations.

#### CUDA implementation total program duration:
```sh
$ nvcc lg_cuda.cu -o lg_cuda
$ ./lg_cuda
Elapsed time: 0.049655 seconds
Theta after gradient descent:
0.125907 0.151487 0.134644
```
#### Cupy implementation total program duration:
```sh
$ python lg_cupy.py
Theta after gradient descent: [[0.06212644]
 [0.12514137]
 [0.09668753]]
Elapsed time: 0.24989748001098633 seconds
```
#### Numba implementation total program duration:
```sh
$ python lg_numba.py
Theta after gradient descent: [[0.06453408]
 [0.12351934]
 [0.09929141]]
Elapsed time: 0.36286187171936035 seconds
```
In addition to comparing the total duration of executing the identical programs, there is a need to compare the specific kernel launches with identical launch configurations and workload as well. 

#### SOL Table for CUDA
```sh
$ ncu ./lg_cuda

compute_cost_kernel(const float *, const float *, float *, int) (157, 1, 1)x(64, 1, 1), Context 1, Stream 7, Device 6, CC 8.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.53
    SM Frequency            cycle/nsecond         1.09
    Elapsed Cycles                  cycle        4,760
    Memory Throughput                   %         2.05
    DRAM Throughput                     %         0.94
    Duration                      usecond         4.35
    L1/TEX Cache Throughput             %         4.19
    L2 Cache Throughput                 %         1.47
    SM Active Cycles                cycle     2,326.82
    Compute (SM) Throughput             %         2.57
    ----------------------- ------------- ------------
```

#### SOL Table for Cupy
```sh
$ ncu --nvtx --nvtx-include "CUDAkernel/" python lg_cupy.py

Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         3.14
    SM Frequency            cycle/nsecond         2.25
    Elapsed Cycles                  cycle       10,217
    Memory Throughput                   %         0.95
    DRAM Throughput                     %         0.44
    Duration                      usecond         4.54
    L1/TEX Cache Throughput             %         3.69
    L2 Cache Throughput                 %         0.66
    SM Active Cycles                cycle     2,636.04
    Compute (SM) Throughput             %         1.32
    ----------------------- ------------- ------------
```
#### SOL Table for Numba
```sh
$ ncu --nvtx --nvtx-include "CUDAkernel/" python lg_numba.py

Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         2.35
    SM Frequency            cycle/nsecond         1.68
    Elapsed Cycles                  cycle        9,131
    Memory Throughput                   %         1.07
    DRAM Throughput                     %         0.49
    Duration                      usecond         5.41
    L1/TEX Cache Throughput             %         3.03
    L2 Cache Throughput                 %         0.72
    SM Active Cycles                cycle     3,209.05
    Compute (SM) Throughput             %         2.63
    ----------------------- ------------- ------------
```

### TakeAways:
In terms of total execution time, Cupy seems to be second right after CUDA, which is the native programming environment for NVIDIA GPUs. However, as the goal of this project is to look for multiple GPU usage for Machine Learning methods, CUDA lacks the necessary abstractions to code complicated ML models with efficiency. Hence, it is seen that Cupy is the best out of the frameworks considered for this purpose. In total duration and kernel duration, Numba is seen to lag.  


## [Part-2: Cupy vs Pytorch](./Part-2)
When GPU-aligned language-inherent methods are used Cupy and Pytorch are observed to show similar performance. For example in the implementation of logistic regression, the Torch  arrays (in Device) and Cupy arrays (in Device) are used. The 2 programs yield similar results in terms of total time taken for execution.

```sh
$ python logistic_cupy.py
Theta after gradient descent: [[-0.29748608]
 [ 0.58119898]
 [ 0.61632247]]
Elapsed time: 0.6893346309661865 seconds

$ python logistic_pytorch.py
Theta after gradient descent: tensor([[-0.3796],
        [ 0.5992],
        [ 0.6117]], device='cuda:6')
Elapsed time: 0.6835305690765381 seconds
```
However, when the 3rd degree polynomial approximation for a sin function is used, Torch arrays are seen to perform better than the Cupy arrays.

```sh
$ python fit_poly_torch.py
Average elapsed time over 20 runs: 0.42211618423461916 seconds

$ python fit_poly_cupy.py
Average elapsed time over 20 runs: 0.8112529397010804 seconds
```
**!!** There is a need for further exploration here, especially because Pytorch shows similar or possibly better performance compared to Cupy in some cases. However the test cases here do not consider the kernel configuration setup for Pytorch because of the aforementioned problems in launching Pybind11 for that purpose. 
