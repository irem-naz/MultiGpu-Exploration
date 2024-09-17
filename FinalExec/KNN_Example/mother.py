import subprocess
import sys

def run_python_script(script_name, gpu_n, fraction):
    # Run the Python script and pass GPU_N and fraction as command-line arguments
    subprocess.run([sys.executable, script_name, str(gpu_n), str(fraction)], check=True)

def run_cuda_script(cu_file, output_binary, gpu_n, fraction):
    # Compile the .cu file using nvcc and create an executable
    compile_command = ['nvcc', cu_file, '-o', output_binary]
    compile_process = subprocess.run(compile_command, check=True)

    # Run the compiled binary
    run_command = ['./' + output_binary]
    subprocess.run(run_command + [str(gpu_n), str(fraction)], check=True)

def run_all_scripts(gpu_n, fraction):
    scripts = [
        "cupyFile.py",
        "numpyFile.py",
        "hybridFile.py",
        "cudaFile.cu"
    ]

    for script in scripts:
        print(f"Running {script} with GPU_N={gpu_n} and fraction={fraction}")

        if script.endswith(".py"):
            run_python_script(script, gpu_n, fraction)
        elif script.endswith(".cu"):
            output_binary = "cuda_program"
            run_cuda_script(script, output_binary, gpu_n, fraction)
        else:
            print(f"Error: Unsupported file type for {script}. Skipping.")
            continue

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mother.py <script_name> [<GPU_N> <fraction>]\nUsage: python mother.py <GPU_N> <fraction>")
        sys.exit(1)

    if len(sys.argv) == 3:
        gpu_n = int(sys.argv[1])
        fraction = float(sys.argv[2])
        run_all_scripts(gpu_n, fraction)
    else:
        script_name = sys.argv[1]
        # Check if GPU_N and fraction are provided
        if len(sys.argv) >= 4:
            gpu_n = int(sys.argv[2])
            fraction = float(sys.argv[3])
        else:
            gpu_n = 1
            fraction = 1.0

        # If it's a Python script, pass GPU_N and fraction
        if script_name.endswith(".py"):
            if len(sys.argv) != 4:
                print("Usage for Python script: python mother_script.py <python_script> <GPU_N> <fraction>")
                sys.exit(1)
            run_python_script(script_name, gpu_n, fraction)

        # If it's a CUDA (.cu) file, compile and run it
        elif script_name.endswith(".cu"):
            output_binary = "cuda_program"
            run_cuda_script(script_name, output_binary, gpu_n, fraction)

        else:
            print("Error: Unsupported file type. Please provide a .py or .cu file.")
            sys.exit(1)
