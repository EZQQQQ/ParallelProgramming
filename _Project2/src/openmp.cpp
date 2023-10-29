//
// OpenMp + SIMD + Reordering Matrix Multiplication
//

#include <immintrin.h>
#include <omp.h> 
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include "mpi.h"

void matrix_multiply_openmp(const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int startRow, int endRow) {
    // Ensure that the dimensions are compatible for multiplication
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = endRow - startRow; // Number of rows to process
    size_t K = matrix1.getCols();
    size_t N = matrix2.getCols();

    size_t BLOCK_SIZE = 128;

    // Iterate over the specified range of rows
    for (size_t i = startRow; i < endRow; i++) {
        for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
            for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
                for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, N); j += 8) {
                    __m256i sum = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result[i][j]));
                    for (size_t k = kk; k < std::min(kk + BLOCK_SIZE, K); ++k) {
                        __m256i multiplier = _mm256_set1_epi32(matrix1[i][k]);
                        __m256i b1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&matrix2[k][j]));
                        __m256i product1 = _mm256_mullo_epi32(multiplier, b1);

                        sum = _mm256_add_epi32(sum, product1);
                    }
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i][j]), sum);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    // MPI Initialization
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Verify input argument format on the root process (rank 0)
    if (world_rank == 0 && argc != 5) {
        std::cerr << "Invalid argument, should be: ./executable thread_num /path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (argc != 5) {
        MPI_Finalize();
        return 1;
    }

    int thread_num = atoi(argv[1);
    omp_set_num_threads(thread_num);

    const std::string matrix1_path = argv[2];
    const std::string matrix2_path = argv[3];
    const std::string result_path = argv[4];

    // Load matrices on the root process and broadcast to other processes
    Matrix matrix1, matrix2, result;
    if (world_rank == 0) {
        matrix1 = Matrix::loadFromFile(matrix1_path);
        matrix2 = Matrix::loadFromFile(matrix2_path);
        result = Matrix(matrix1.getRows(), matrix2.getCols());
    }

    // Broadcast matrix dimensions to all processes
    int M, K, N;
    if (world_rank == 0) {
        M = matrix1.getRows();
        K = matrix1.getCols();
        N = matrix2.getCols();
    }
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine the range of rows to be processed by each process
    int rows_per_process = M / world_size;
    int start_row = world_rank * rows_per_process;
    int end_row = (world_rank == world_size - 1) ? M : start_row + rows_per_process;

    // Perform matrix multiplication using OpenMP and SIMD intrinsics
    auto start_time = std::chrono::high_resolution_clock::now();
    matrix_multiply_openmp(matrix1, matrix2, result, start_row, end_row);
    auto end_time = std::chrono::high_resolution_clock::now();

    // Gather results from all processes to the root process
    Matrix finalResult(M, N);
    MPI_Gather(result[start_row], M * N / world_size, MPI_INT, finalResult[start_row], M * N / world_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Save the result on the root process
    if (world_rank == 0) {
        finalResult.saveToFile(result_path);
        std::cout << "Output file to: " << result_path << std::endl;
        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " milliseconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
