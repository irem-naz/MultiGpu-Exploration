import cupy as cp
import cupyx
import numpy as np
import time


class TGPUplan:
    def __init__(self):
        self.dataN = 0
        self.h_Data = None
        self.h_Sum = None
        self.d_Data = None
        self.d_Sum = None
        self.h_Sum_from_device = None
        self.stream = None

# CUDA kernel for reduction
reduce_kernel = cp.RawKernel(r'''
extern "C" __global__
void reduceKernel(float *d_Result, float *d_Input, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadN = gridDim.x * blockDim.x;
    float sum = 0;

    for (int pos = tid; pos < N; pos += threadN)
        sum += d_Input[pos];

    d_Result[tid] = sum;
}
''', 'reduceKernel')

# Function to initialize the GPU plan
def initialize_plan(plan, GPU_N, DATA_N):
    for i in range(GPU_N):
        plan[i].dataN = DATA_N // GPU_N

    for i in range(DATA_N % GPU_N):
        plan[i].dataN += 1

    gpuBase = 0
    for i in range(GPU_N):
        plan[i].h_Sum = np.zeros(1, dtype=cp.float32)
        gpuBase += plan[i].dataN

    for i in range(GPU_N):
        with cp.cuda.Device(i):
            plan[i].stream = cp.cuda.Stream(non_blocking = True)
            with plan[i].stream:
                # On device, input and output
                plan[i].d_Data = cp.zeros(plan[i].dataN, dtype=cp.float32)
                plan[i].d_Sum = cp.zeros(BLOCK_N * THREAD_N, dtype=cp.float32)
                
                # The following imitates cudaMallocHost, page-locked/pinned host memory 
                # for future async mem transfers

                # Allocate pinned memory for h_Sum_from_device
                plan[i].h_Sum_from_device = cupyx.empty_pinned((BLOCK_N * THREAD_N,), dtype=cp.float32)
                # plan[i].h_Sum_from_device = np.empty((BLOCK_N * THREAD_N,), dtype=cp.float32)
                
                # Allocate pinned memory for h_Data
                plan[i].h_Data = cupyx.empty_pinned((plan[i].dataN,), dtype=cp.float32)
                # plan[i].h_Data = np.empty((plan[i].dataN,), dtype=cp.float32)
                
                # Initialize h_Data with random values
                temp_data = cp.random.rand(plan[i].dataN).astype(cp.float32)
                temp_data.get(stream=plan[i].stream, out=plan[i].h_Data, blocking=False)
               
# Constants
MAX_GPU_COUNT = 32
DATA_N = 1048576 * 32 
BLOCK_N = 32
THREAD_N = 256
ACCUM_N = BLOCK_N * THREAD_N

def main():
    print("Starting simpleMultiGPU")
    GPU_N = cp.cuda.runtime.getDeviceCount()

    # initialize structs per GPU
    plan = [TGPUplan() for _ in range(GPU_N)]
    # return values will be stored here per GPU
    h_SumGPU = cp.zeros(GPU_N, dtype=cp.float32)


    if GPU_N > MAX_GPU_COUNT:
        GPU_N = MAX_GPU_COUNT

    print(f"CUDA-capable device count: {GPU_N}")
    print("Generating input data...\n")

    initialize_plan(plan, GPU_N, DATA_N)

    # Start timing and compute on GPU(s)
    print(f"Computing with {GPU_N} GPUs...")
    start_time = time.time()

    # Copy data to GPU, launch the kernel and copy data back. All asynchronously
    for i in range(GPU_N):
        with cp.cuda.Device(i):
            with plan[i].stream:
                plan[i].d_Data.set(plan[i].h_Data, stream=plan[i].stream)
                # Commented out in V3
                # plan[i].stream.synchronize()


    for i in range(GPU_N):
        with cp.cuda.Device(i):
            with plan[i].stream:
                reduce_kernel((BLOCK_N,), (THREAD_N,), (plan[i].d_Sum, plan[i].d_Data, plan[i].dataN), stream=plan[i].stream)
                plan[i].d_Sum.get(stream=plan[i].stream, out=plan[i].h_Sum_from_device, blocking=False)

    # Process GPU results
    sumGPU = 0.0
    for i in range(GPU_N):
        plan[i].stream.synchronize()
        sum = np.sum(plan[i].h_Sum_from_device)
        h_SumGPU[i] = sum
        sumGPU += sum

    gpu_time = (time.time() - start_time) * 1000
    print(f"  GPU Processing time: {gpu_time} (ms)\n")

    # Compute on Host CPU
    print("Computing with Host CPU...\n")
    start_time = time.time()
    sumCPU = 0.0

    for i in range(GPU_N):
        sumCPU += np.sum(plan[i].h_Data)

    cpu_time = (time.time() - start_time) * 1000
    print(f"  CPU Processing time: {cpu_time} (ms)\n")

    # Compare GPU and CPU results
    print("Comparing GPU and Host CPU results...")
    diff = abs(sumCPU - sumGPU) / abs(sumCPU)
    print(f"  GPU sum: {sumGPU}\n  CPU sum: {sumCPU}")
    print(f"  Relative difference: {diff:E}\n")

    if diff < 1e-5:
        print("Results match.")
    else:
        print("Results do not match!")

if __name__ == "__main__":
    main()
