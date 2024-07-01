# Comparing Frameworks

There are 3 frameworks in Python language that allows low-level kernel configuration management: Cupy - a drop-in replacement for Numpy -, Numba - which is JIT compiler for CUDA option -, and Pytorch that can launch its arrays in GPU due to inherent GPU acceleration and allows kernel management through C to Python translation libraries such as Pybind11.

The same binary logistic regression is implemented for each of the models.

## Part-1: CUDA vs Cupy vs Numba
**!** Since there were many problems faced in implementing Pybind11, Pytorch is not compared to other frameworks here that use CUDA kernels to manage kernel launch configurations.

#### CUDA implementation total program duration:
```sh
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
In addition to comparing the total duration of executing the identical programs, there is a need to compare the specific kernel launches with identical launch configurations and workload. In this part CUDA is not consi

#### SOL Table for Cupy
```sh
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
## Part-2: Cupy vs Pytorch
