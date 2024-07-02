# MultiGpu-Exploration
This is a project that explores multiple GPU usage in ML with low level thread and block management enabled.

## Getting Started
In order to run the project the following dependencies are needed. Conda environment is recommended for ease of installing the libraries, both Miniconda and Anaconda works.
 - Python
 - CUDA & C compiler
 ### Libraries used:
 - Numpy
 - Nvtx
 - CuPy
 - Numba
   
They all can be installed using the environment.yml in this repository using the following command:
```sh
conda env create -f environment.yml
```

## Event History
This will serve as a log for all the actions taken on the project and the decisions made.

#### Until the creation of this log file following has been done:
✔️ [Step 1](Step%201/README.md) CUDA exploration with Arithmetic.cu by implementing a simple kernel that does id calculation and stores it as value.
   - implemented using CUDA
     
✔️ [Step 1](Step%201/README.md) CuPy exploration with Arithmetic.py by implementing a simple kernel that does id calculation and stores it as value.
   - implemented using CuPy, with the same kernel configuration to the CUDA version, and compared between models' performance.
     
✔️ [Step 2](Step%202/README.md) Using CuPy for Logistic Regression
   - implemented using CuPy and Numpy for performance comparison
     
✔️ [Step 3](Step%203/README.md) Comparing Frameworks: CuPy, Numba, Pytorch.
   Relevant links for implementation: [CuPy](https://docs.cupy.dev/en/stable/user_guide/kernel.html), [Numba](https://numba.pydata.org/numba-doc/latest/cuda/kernels.html), [Pytorch](https://pytorch.org/tutorials/advanced/cpp_extension.html).
   - implemented successfully for only CuPy and Numba.
     
#### After the creation of this log file following is being done:
❌ [Step 4](Step%204/README.md) Using KNN for Cupy Multi-GPU implementation with MNIST dataset
   - implemented using KNN for Cupy Multi-GPU implementation with MNIST dataset.
     - ✔️ Initially implemented it in pure CuPy and ran it.
     - Converted the euclidean distance calculation into CUDA kernel to disperse between GPUs.
       - **Problem:** Despite implementing rawKernel, and establishing streams and GPU contexts, each action is processed sequentially.
       - **Goal** is to gain any type of concurrency between GPUs.
       - [Follow the updates on the implementation here.](Step%204/README.md)
   
