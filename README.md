# MultiGpu-Exploration
This is a project that explores multiple GPU usage in ML with low level thread and block management enabled.

## Getting Started
In order to run the project the following dependencies are needed. Conda environment is recommended for ease of installing the libraries, both Miniconda and Anaconda work.
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

## Purpose of the README file
This will serve as a log for all the actions taken on the project and the decisions made.

#### Until the creation of this log file following has been done:
✔️ CUDA exploration with Arithmetic.cu by implementing a simple kernel that does id calculation and stores it as value.
   - implemented using CUDA
     
✔️ CuPy exploration with Arithmetic.py by implementing a simple kernel that does id calculation and stores it as value.
   - implemented using CuPy, with the same kernel configuration to the CUDA version, and compared between models' performance.
       ##### Observed Outputs:
     
✔️ Using CuPy for Logistic Regression
   - implemented using CuPy and Numpy for performance comparison
       ##### Observed Outputs:
     
✔️ Comparing Frameworks: CuPy, Numba, Pytorch
   Relevant links for implementation: [CuPy](https://docs.cupy.dev/en/stable/user_guide/kernel.html), [Numba](https://numba.pydata.org/numba-doc/latest/cuda/kernels.html), [Pytorch](https://pytorch.org/tutorials/advanced/cpp_extension.html).
   - implemented successfully for only CuPy and Numba.
       ##### Observed Outputs:
     
#### After the creation of this log file following is being done:
❌ Using KNN for Cupy Multi-GPU implementation with MNIST dataset
   - implemented using KNN for Cupy Multi-GPU implementation with MNIST dataset.
     - ✔️ Initially implemented it in pure CuPy and ran it.
     - Converted the euclidean distance calculation into CUDA kernel to disperse between GPUs.
       - **Problem**  is that because of the direct conversion of the algorithm from pure CuPy implementation to RawKernel, memory transfers and other classification calculations come between euclidean distance calculations.
       - **Goal** is to have all the euclidean distance calculations run first and parallelize that between GPUs, before memory transfers and classification calculations take place.


   
