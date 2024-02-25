# ParallelProgramming

# CMake Configuration

1. **Configuration:**
   - Use CMake to set up the project's build environment.
   - Navigate to the project directory (`../project1/build/`).
   - Run `bash cmake ..` to generate necessary build files.

2. **Compilation:**
   - Compile the project using the `make` command.
   - Use parallel jobs with `bash make -j4` to speed up compilation.
   - Links libraries and generates the executable binary.

3. **Cluster Submission Script:**
   - Utilize a submission script to run the program on the cluster.
   - Example using `sbatch` command: `sh src/scripts/sbatch.sh & sh src/scripts/sbatch_bonus.sh`.
   - Specifies job parameters like CPU cores, memory requirements, and the executable name.
   - Responsible for launching the program on cluster nodes with specified configurations.

# High-Level Overview of Project 1: Image Processing with Parallel Programming Models
- Part A: RGB to Grayscale
- Part B: RGB to Smooth Filtering

## Parallel Programming Models:
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

# High-Level Overview of Project 2: Matrix Multiplication

## Parallel Programming Models: Memory Locality

## Blocking (Tiling):
- **Description:**
  - Divides matrices into smaller blocks (tiles) for cache locality.
  - Promotes cache locality and reduces cache misses.

## Data-Level Parallelism:

### Blocking (Tiling):
- **Description:**
  - Dynamically adjusts `BLOCK_SIZE` based on matrix dimensions.

- **Change Loop Order:**
  - Uses SIMD instructions for parallelizing matrix multiplication.

## Thread-Level Parallelism:

### Blocking (Tiling):
- **Description:**
  - Uses OpenMP directives for outer loop parallelization.

- **Change Loop Order:**
  - OpenMP directives parallelize computation within each block.

## Process-Level Parallelism:

### Blocking (Tiling):
- **Description:**
  - MPI processes dynamically balance load.

- **Change Loop Order:**
  - Uses MPI processes for parallel execution.

## CUDA Kernel for Matrix Multiplication:
- **Description:**
  - Offloads matrix multiplication to GPU using CUDA kernel.
  - Utilizes shared memory for storing submatrices.

# High-Level Overview of Project 3: Sorting Algorithm

## Task 1: Parallel Quick Sort with MPI

### Parallelization:
- Uses MPI for sorting across multiple processes.
- Each process performs a local quicksort on a vector segment.

### Key Functions:
- `partition`: Divides a vector based on a pivot.
- `sequentialquickSort`: Implements sequential quicksort.
- `quickSort`: Distributes the vector among MPI processes, performs parallel quicksort, and merges sorted segments.

### Strategy:
- MPI processes handle local sorting.
- Results are merged on the master process.

## Task 2: Parallel Bucket Sort with MPI

### Parallelization:
- Each process performs bucket sort on its data.
- Sorted buckets are gathered and merged on the master process.

### Bucket Sort Function:
- `bucketSort`: Distributes elements into local buckets, sorts each using insertion sort, gathers bucket size info, and reconstructs the sorted vector.

### Strategy:
- Independent sorting in buckets.
- Merging sorted buckets reconstructs the final vector.

## Task 3: Parallel Odd-Even Sort

### Odd-Even Sorting:
- Each process sorts its local array using sequential odd-even sort.
- Odd and even phases involve communication and merging with neighboring processes.

### Functions:
- `oddEvenSort`: Orchestrates parallel odd-even sort using MPI.
- `oddEvenIteration`: One iteration of parallel odd-even transposition sort.
- `mergeSplitLow` and `mergeSplitHigh`: Merges smallest and largest elements.

### Strategy:
- Independent sorting with exchange and merge.
- Results are gathered for broadcasting.

## Task 5: Dynamic Thread-Level Parallel Quick Sort (OpenMP Tasking)

### OpenMP Parallelization:
- Uses OpenMP directives for parallel quicksort.

### Features:
- #pragma directives for parallel region and task creation.
- Task synchronization with #pragma omp taskwait.
- Falls back to sequential quicksort for small arrays.

### Strategy:
- Concurrent task execution using multiple threads.
- Utilizes OpenMP for multicore processors.

# High-Level Overview of Project 4: Machine Learning (Neural Network Training)

## Task 1: Train MNIST with Softmax Regression

### Purpose:
- Train a softmax regression model on MNIST dataset.
- Functions handle batch processing, one-hot encoding, softmax normalization, gradient computation, and evaluation.

### Key Functions:
- `softmax_regression_epoch_cpp`: Single training epoch with SGD.
- `train_softmax`: Fully trains softmax classifier over multiple epochs.

## Task 2: Accelerate Softmax with OpenACC

### Purpose:
- Utilize OpenACC directives for parallelization and optimization.
- Improve execution time efficiency, particularly on GPUs.

### Key Features:
- Loop parallelization, data management, and data parallelism.
- Independent loop optimization and GPU acceleration.

## Task 3: Train MNIST with Neural Network

### Purpose:
- Implement training and evaluation processes for a two-layer neural network.
- Emphasize simplicity and clarity in code structure.

### Key Functions:
- `nn_epoch_cpp`: Single epoch of SGD for a two-layer neural network.
- `evaluate_nn`: Evaluate network performance on a dataset.
- `train_nn`: Fully train a neural network over multiple epochs.

## Task 4: Accelerate Neural Network with OpenACC

### Purpose:
- Optimize memory usage, minimize data transfer overhead, and parallelize computationally intensive tasks.
- Improve efficiency, especially on parallel architectures like GPUs.

### Key Features:
- Data movement, parallelization of loops, collapse and independent clauses.
- Memory optimization, parallelization of matrix operations, and explicit memory management.
