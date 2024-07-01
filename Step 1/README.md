# Comparing .py vs .cu version of the same code

The simple kernel presented here both stores the id of an array as a value to that location in the array using GPU. 

The duration for both kernel launches is roughly the same with the following SOL tables as follows:
### Pure CUDA:
```sh
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

### CuPy with RawKernel:
```sh
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

