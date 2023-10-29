# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(CUDA_Matrix_Multiplication)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find and include CUDA package
set(CMAKE_CUDA_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4")
enable_language(CUDA)  # 启用CUDA支持
set(CMAKE_CUDA_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/nvcc")
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("CUDA FOUND, Version: " ${CUDA_VERSION_STRING})
endif ()

SET(CMAKE_CXX_COMPILER g++)  

# Set the appropriate standard library version
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -static-libstdc++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")

## CUDA
cuda_add_executable(cuda
        cuda.cu
        ../matrix.cpp ../matrix.hpp)
target_link_libraries(cuda cudart)