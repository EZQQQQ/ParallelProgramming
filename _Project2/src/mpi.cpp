//
// MPI + OpenMp + SIMD + Reordering Matrix Multiplication
//

#include <mpi.h>  // MPI Header
#include <omp.h> 
#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

#define MASTER 0
#define TAG_GATHER 0

// Function to perform matrix multiplication using MPI, OpenMP, and SIMD
Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2, int start, int end) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    Matrix result(M, N);

    size_t BLOCK_SIZE = 64;

    int num_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();

    int rows_per_thread = ((end - start) + num_threads - 1) / num_threads;
    int start_row = thread_id * rows_per_thread;
    int end_row = std::min(start_row + rows_per_thread, int(M));

    for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
        for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
            #pragma omp parallel for collapse(2)
            for (size_t i = start_row; i < end_row; i++) {
                for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, N); j+=8) {
                    __m256i sum = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result[i][j]));
                    for (size_t k = kk; k < std::min(kk + BLOCK_SIZE, K); ++k) {
                        __m256i multiplier = _mm256_set1_epi32(matrix1[i][k]);
                        __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matrix2[k][j]));
                        __m256i product1 = _mm256_mullo_epi32(multiplier, b1);

                        sum = _mm256_add_epi32(sum, product1);
                    }
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i][j]), sum);
                }
            }
        }
    }

    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    // Read Matrix
    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    int rows_per_task = matrix1.getRows() / numtasks;
    int remaining_rows = matrix1.getRows() % numtasks;

    std::vector<int> task_row_splits(numtasks + 1, 0);
    int remaining_rows_left = 0;

    for (int i = 0; i < numtasks; i++) {
        task_row_splits[i + 1] = task_row_splits[i] + rows_per_task + (i < remaining_rows ? 1 : 0);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    if (taskid == MASTER) {
        Matrix result = matrix_multiply_mpi(matrix1, matrix2, task_row_splits[taskid], task_row_splits[taskid+1]);

        // Your Code Here for Synchronization!
        for (int i = MASTER + 1; i < numtasks; i++) {
            int task_start_row = task_row_splits[i];
            int task_row_count = task_row_splits[i + 1] - task_row_splits[i];
            int col_count = matrix2.getCols();

            // Allocate memory for the partial result
            std::vector<int> partial_result(task_row_count * col_count, 0);

            MPI_Recv(partial_result.data(), task_row_count * col_count, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Update the master result matrix with the partial result
            for (int row = 0; row < task_row_count && (row + task_start_row) < matrix1.getRows(); row++) {
                for (int col = 0; col < col_count && col < matrix2.getCols(); col++) {
                    result[row + task_start_row][col] = partial_result[row * col_count + col];
                }
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

        result.saveToFile(result_path);

        std::cout << "Output file to: " << result_path << std::endl;

        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;
    } else {
        // Worker's matrix multiplication
        Matrix result = matrix_multiply_mpi(matrix1, matrix2, task_row_splits[taskid], task_row_splits[taskid + 1]);

        int task_start_row = task_row_splits[taskid];
        int task_row_count = task_row_splits[taskid + 1] - task_row_splits[taskid];
        int col_count = matrix2.getCols();

        // Flatten the partial result into a 1D array
        std::vector<int> partial_result(task_row_count * col_count);

        for (int row = 0; row < task_row_count; row++) {
            for (int col = 0; col < col_count; col++) {
                partial_result[row * col_count + col] = result[row][col];
            }
        }

        // Send the partial result back to the master
        MPI_Send(partial_result.data(), task_row_count * col_count, MPI_INT, MASTER, TAG_GATHER, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}