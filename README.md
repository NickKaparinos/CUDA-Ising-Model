## Cuda-Ising-Model
Calculation of the Ising Model lattice using CUDA


----
# V0
Sequential calculation using the CPU.

# V1
Parallel calculation with one one thread per magnetic moment using CUDA.

# V2
Parallel calculation with one one thread block of magnetic moments using CUDA.

# V3
Parallel calculation with multiple threads reading from the shared memory using CUDA.


----
Each version provides a speedup in execution time compared to the previous version.
