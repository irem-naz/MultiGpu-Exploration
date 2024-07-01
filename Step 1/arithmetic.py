import cupy as cp
import nvtx
import time

# Define the raw kernel
simple_kernel = cp.RawKernel(r'''
extern "C" __global__
void simpleKernel(int *d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_data[idx] = idx;
    }
}
''', 'simpleKernel')

def main():

    start_time = time.time()

    # Set GPU device 3
    cp.cuda.Device(3).use()

    size = 1024
    h_data = cp.empty(size, dtype=cp.int32)
    d_data = cp.zeros(size, dtype=cp.int32)
   

    # Configure the block and grid size
    threads_per_block = 256
    blocks_per_grid = size // threads_per_block

    # Launch the kernel
    with nvtx.annotate("mykernel"):
        simple_kernel((blocks_per_grid,), (threads_per_block,), (d_data, size))


    # Copy data back to host
    h_data = d_data.get()

    print("CUDA program completed successfully.")
    print(h_data)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()
