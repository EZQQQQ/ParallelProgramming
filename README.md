# Parallel Programming

This repository collects four parallel computing coursework projects covering CPU, distributed, and accelerator-based optimization techniques. Across the projects, the work explores how algorithm design changes when targeting SIMD instructions, multithreading, MPI, GPUs, and OpenACC-style acceleration.

## What This Repository Shows

- low-level performance-oriented C and C++ programming
- practical use of MPI, OpenMP, SIMD, CUDA-style thinking, and OpenACC workflows
- benchmarking and experimentation across different parallelization strategies
- translating algorithmic problems into CPU, cluster, and GPU execution models

## Tech Used

- C and C++
- CMake
- MPI
- OpenMP
- SIMD optimization
- OpenACC
- SLURM job scripts

## Projects

### Project 1: Image processing

Focuses on accelerating image-processing tasks such as RGB-to-grayscale conversion and smoothing filters using multiple parallel programming models. The original notes reference SIMD, MPI, OpenMP, Pthreads, CUDA, and OpenACC as the target optimization approaches explored in this part of the coursework.

### Project 2: Matrix multiplication

Explores matrix multiplication through memory-locality and parallel-computing techniques, including:

- blocking and tiling for cache efficiency
- loop-order changes for performance tuning
- SIMD-based data-level parallelism
- OpenMP-based thread-level parallelism
- MPI-based process-level parallelism
- GPU-oriented execution ideas reflected in the performance and submission materials

Relevant source files are in `_Project2/src/`, including implementations such as `naive.cpp`, `locality.cpp`, `simd.cpp`, `openmp.cpp`, and `mpi.cpp`.

### Project 3: Parallel sorting

Implements and analyzes multiple sorting strategies under parallel execution models, including:

- MPI-based quicksort
- MPI-based bucket sort
- odd-even transposition sort with inter-process communication
- OpenMP task-based quicksort

This project emphasizes workload distribution, merge strategies, and synchronization between parallel workers.

### Project 4: Neural network training

Focuses on machine learning workloads optimized for parallel systems using MNIST-style training tasks. The project includes:

- softmax regression training
- two-layer neural network training
- OpenACC-accelerated softmax and neural network variants
- SLURM scripts for cluster and GPU execution

Key source files are in `_Project4/src/`, and the repository includes job scripts such as `_Project4/sbatch.sh` and `_Project4/test.sh`.

## Repository Layout

- `_Project1/`: image-processing project
- `_Project2/`: matrix multiplication project with source, sample matrices, and performance data
- `_Project3/`: sorting project with source and submission assets
- `_Project4/`: neural network training project with cluster scripts and accelerator-oriented implementations

## Build and Run

Each project is organized as its own CMake-based unit.

Typical local workflow:

```bash
cd _Project2
mkdir -p build
cd build
cmake ..
make -j4
```

For cluster-based runs, use the included SLURM scripts where available, for example:

```bash
cd _Project4
sbatch sbatch.sh
```

## Notes

- the repository includes both source code and coursework submission materials
- some project folders contain generated build files, reports, and archived submission artifacts
- this repo is best read as a record of hands-on systems and high-performance computing coursework rather than a single standalone application
