# ParallelProgramming

# High-Level Overview of Project 1: Image Processing with Parallel Programming Models
- Part A: RGB to Grayscale
- Part B: RGB to Smooth Filtering

## 1. Project Setup:
- Utilizes CMake for project configuration.
- Compilation and execution managed through scripts.
- Cluster deployment using MPI with specified configurations.

## 2. Parallel Programming Models:
- **SIMD (Single Instruction, Multiple Data):**
  - Processes multiple data elements simultaneously.
  - Utilizes AVX2 and AVX-512 instructions for vectorized operations.

- **MPI (Message Passing Interface):**
  - Enables parallelism with distributed memory across cluster nodes.
  - Uses message passing for communication between processes.

- **OpenMP:**
  - Facilitates parallelism in shared-memory systems.
  - Allows developers to specify parallel regions and constructs.

- **Pthreads (POSIX Threads):**
  - Low-level threading library for task-level parallelism.
  - Requires manual thread creation and management.

- **CUDA (Compute Unified Device Architecture):**
  - Parallel computing platform for GPU processing.
  - Employs data-level and thread-level parallelism.

- **OpenACC:**
  - API for parallel programming, simplifying GPU code parallelization.
  - Uses directives for specifying parallel regions and data parallelism.
