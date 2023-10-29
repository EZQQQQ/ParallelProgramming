//
// Matrix Multiplication with CUDA, for bonus
//
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include "../matrix.hpp"

const int TILE_SIZE = 32;

__global__ void matrixMultiply(float* A, float* B, float* C, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float subTileA[TILE_SIZE][TILE_SIZE];
    __shared__ float subTileB[TILE_SIZE][TILE_SIZE];

    float Cvalue = 0.0;

    for (int t = 0; t < K / TILE_SIZE; t++) {
        subTileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        subTileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            Cvalue += subTileA[threadIdx.y][i] * subTileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = Cvalue;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable "
                  << "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result" << std::endl;
        return 1;
    }

    const std::string matrix1_path = argv[1];
    const std::string matrix2_path = argv[2];
    const std::string result_path = argv[3];

    // Read matrix dimensions from your matrix objects
    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);
    int N = matrix2.getCols();
    int M = matrix1.getRows();
    int K = matrix1.getCols();

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = new float[N * N];
    h_B = new float[N * N];
    h_C = new float[N * N];

    // Load matrix data from your matrix objects into host arrays
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            h_A[i * K + j] = matrix1[i][j];
        }
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            h_B[i * N + j] = matrix2[i][j];
        }
    }

    // Allocate device memory for A, B, and C
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim(N / blockDim.x, N / blockDim.y);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch the matrix multiplication kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, M, K);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Copy the result back from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Save the result matrix to a file
    std::ofstream result_file(result_path, std::ios::binary);
    if (!result_file.is_open()) {
        std::cerr << "Error: Unable to open result file." << std::endl;
        return 1;
    }
    
    for (int i = 0; i < N * N; ++i) {
        result_file.write(reinterpret_cast<char*>(&h_C[i]), sizeof(float));
    }

    std::cout << "Output file to: " << result_path << std::endl;
    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds" << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
