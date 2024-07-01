# Comparing CUDA and Cupy versions of Simple Kernel

The simple kernel presented here stores the id of an array as a value to that location in the array using GPU. There are 2 implementations curated here, one using only CUDA and another with Cupy methods as well as a CUDA RawKernel. Kernel launch configurations are identical for comparisons, they can be modified to observe the affect of different configurations for GPU utilization.

The duration for both kernel launches is roughly the same with the following SOL tables as follows:

### Pure CUDA:
```sh
$ nvcc arithmetic.cu -o arithmetic
$ ncu ./arithmetic

simpleKernel(int *, int) (1, 1, 1)x(1024, 1, 1), Context 1, Stream 7, Device 3, CC 8.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.04
    SM Frequency            cycle/usecond       739.66
    Elapsed Cycles                  cycle        3,078
    Memory Throughput                   %         0.54
    DRAM Throughput                     %         0.00
    Duration                      usecond         4.16
    L1/TEX Cache Throughput             %        60.87
    L2 Cache Throughput                 %         0.54
    SM Active Cycles                cycle         8.52
    Compute (SM) Throughput             %         0.04
    ----------------------- ------------- ------------
```

### Cupy with RawKernel:
```sh
$ ncu python arithmetic.py

simpleKernel (1, 1, 1)x(1024, 1, 1), Context 1, Stream 7, Device 3, CC 8.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.05
    SM Frequency            cycle/usecond       748.83
    Elapsed Cycles                  cycle        3,119
    Memory Throughput                   %         0.82
    DRAM Throughput                     %         0.00
    Duration                      usecond         4.16
    L1/TEX Cache Throughput             %        60.54
    L2 Cache Throughput                 %         0.82
    SM Active Cycles                cycle         8.56
    Compute (SM) Throughput             %         0.04
    ----------------------- ------------- ------------
```

